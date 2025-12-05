// GPU Direct Storage (GDS) / cuFile bindings
// Provides direct file-to-GPU DMA for faster weight loading
// Falls back gracefully when GDS is not available

#include "binfer.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cerrno>

// cuFile types and function pointers
// We dynamically load these to avoid hard dependency on libcufile

typedef void* CUfileHandle_t;

typedef enum {
    CU_FILE_SUCCESS = 0,
    CU_FILE_DRIVER_NOT_INITIALIZED = 5001,
} CUfileOpError;

typedef struct {
    CUfileOpError err;
    int cu_err;
} CUfileError_t;

typedef enum {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
} CUfileFileHandleType;

typedef struct {
    CUfileFileHandleType type;
    union {
        int fd;
        void* handle;
    } handle;
    void* fs_ops;
} CUfileDescr_t;

// Function pointer types
typedef CUfileError_t (*cuFileDriverOpen_t)(void);
typedef CUfileError_t (*cuFileDriverClose_t)(void);
typedef CUfileError_t (*cuFileHandleRegister_t)(CUfileHandle_t*, CUfileDescr_t*);
typedef void (*cuFileHandleDeregister_t)(CUfileHandle_t);
typedef CUfileError_t (*cuFileBufRegister_t)(const void*, size_t, int);
typedef CUfileError_t (*cuFileBufDeregister_t)(const void*);
typedef ssize_t (*cuFileRead_t)(CUfileHandle_t, void*, size_t, off_t, off_t);

// Global state
static void* g_cufile_lib = nullptr;
static bool g_gds_available = false;
static bool g_gds_initialized = false;

// Function pointers
static cuFileDriverOpen_t fn_cuFileDriverOpen = nullptr;
static cuFileDriverClose_t fn_cuFileDriverClose = nullptr;
static cuFileHandleRegister_t fn_cuFileHandleRegister = nullptr;
static cuFileHandleDeregister_t fn_cuFileHandleDeregister = nullptr;
static cuFileBufRegister_t fn_cuFileBufRegister = nullptr;
static cuFileBufDeregister_t fn_cuFileBufDeregister = nullptr;
static cuFileRead_t fn_cuFileRead = nullptr;

// Helper to load cuFile library
static bool load_cufile_library() {
    if (g_cufile_lib) return true;

    // Try common paths
    const char* paths[] = {
        "libcufile.so",
        "libcufile.so.0",
        "/usr/local/cuda/lib64/libcufile.so",
        "/usr/lib/x86_64-linux-gnu/libcufile.so",
        nullptr
    };

    for (const char** path = paths; *path; ++path) {
        g_cufile_lib = dlopen(*path, RTLD_LAZY);
        if (g_cufile_lib) break;
    }

    if (!g_cufile_lib) {
        return false;
    }

    // Load function pointers
    fn_cuFileDriverOpen = (cuFileDriverOpen_t)dlsym(g_cufile_lib, "cuFileDriverOpen");
    fn_cuFileDriverClose = (cuFileDriverClose_t)dlsym(g_cufile_lib, "cuFileDriverClose");
    fn_cuFileHandleRegister = (cuFileHandleRegister_t)dlsym(g_cufile_lib, "cuFileHandleRegister");
    fn_cuFileHandleDeregister = (cuFileHandleDeregister_t)dlsym(g_cufile_lib, "cuFileHandleDeregister");
    fn_cuFileBufRegister = (cuFileBufRegister_t)dlsym(g_cufile_lib, "cuFileBufRegister");
    fn_cuFileBufDeregister = (cuFileBufDeregister_t)dlsym(g_cufile_lib, "cuFileBufDeregister");
    fn_cuFileRead = (cuFileRead_t)dlsym(g_cufile_lib, "cuFileRead");

    // Check all required functions loaded
    if (!fn_cuFileDriverOpen || !fn_cuFileDriverClose ||
        !fn_cuFileHandleRegister || !fn_cuFileHandleDeregister ||
        !fn_cuFileBufRegister || !fn_cuFileBufDeregister || !fn_cuFileRead) {
        dlclose(g_cufile_lib);
        g_cufile_lib = nullptr;
        return false;
    }

    return true;
}

// Wrapper struct for file handle + fd
struct GDSFileHandle {
    CUfileHandle_t cufile_handle;
    int fd;
};

extern "C" {

int binfer_gds_available() {
    if (!load_cufile_library()) {
        return 0;
    }

    // Try to open driver to check if GDS is actually usable
    CUfileError_t err = fn_cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        return 0;
    }

    fn_cuFileDriverClose();
    g_gds_available = true;
    return 1;
}

BinferError binfer_gds_init() {
    if (g_gds_initialized) {
        return BINFER_SUCCESS;
    }

    if (!load_cufile_library()) {
        return BINFER_ERROR_CUDA;
    }

    CUfileError_t err = fn_cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        return BINFER_ERROR_CUDA;
    }

    g_gds_initialized = true;
    g_gds_available = true;
    return BINFER_SUCCESS;
}

BinferError binfer_gds_close() {
    if (!g_gds_initialized) {
        return BINFER_SUCCESS;
    }

    if (fn_cuFileDriverClose) {
        fn_cuFileDriverClose();
    }

    g_gds_initialized = false;
    return BINFER_SUCCESS;
}

BinferError binfer_gds_register_buffer(void* buffer, size_t size) {
    if (!g_gds_initialized) {
        return BINFER_ERROR_CUDA;
    }

    CUfileError_t err = fn_cuFileBufRegister(buffer, size, 0);
    if (err.err != CU_FILE_SUCCESS) {
        return BINFER_ERROR_CUDA;
    }

    return BINFER_SUCCESS;
}

BinferError binfer_gds_deregister_buffer(void* buffer) {
    if (!g_gds_initialized) {
        return BINFER_ERROR_CUDA;
    }

    CUfileError_t err = fn_cuFileBufDeregister(buffer);
    if (err.err != CU_FILE_SUCCESS) {
        return BINFER_ERROR_CUDA;
    }

    return BINFER_SUCCESS;
}

BinferError binfer_gds_open(const char* path, void** handle) {
    if (!g_gds_initialized) {
        *handle = nullptr;
        return BINFER_ERROR_CUDA;
    }

    // Open file with O_DIRECT for GDS
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        *handle = nullptr;
        return BINFER_ERROR_CUDA;
    }

    // Create and register cuFile handle
    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    descr.fs_ops = nullptr;

    GDSFileHandle* gds_handle = new GDSFileHandle();
    gds_handle->fd = fd;

    CUfileError_t err = fn_cuFileHandleRegister(&gds_handle->cufile_handle, &descr);
    if (err.err != CU_FILE_SUCCESS) {
        close(fd);
        delete gds_handle;
        *handle = nullptr;
        return BINFER_ERROR_CUDA;
    }

    *handle = gds_handle;
    return BINFER_SUCCESS;
}

BinferError binfer_gds_close_file(void* handle) {
    if (!handle) {
        return BINFER_SUCCESS;
    }

    GDSFileHandle* gds_handle = (GDSFileHandle*)handle;

    if (fn_cuFileHandleDeregister) {
        fn_cuFileHandleDeregister(gds_handle->cufile_handle);
    }

    close(gds_handle->fd);
    delete gds_handle;

    return BINFER_SUCCESS;
}

ssize_t binfer_gds_read(
    void* handle,
    void* gpu_buffer,
    size_t size,
    size_t file_offset,
    size_t buffer_offset
) {
    if (!handle || !g_gds_initialized) {
        return -1;
    }

    GDSFileHandle* gds_handle = (GDSFileHandle*)handle;

    ssize_t bytes_read = fn_cuFileRead(
        gds_handle->cufile_handle,
        gpu_buffer,
        size,
        (off_t)file_offset,
        (off_t)buffer_offset
    );

    return bytes_read;
}

} // extern "C"

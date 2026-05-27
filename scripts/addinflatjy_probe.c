/*
 * addinflatjy_probe.c - isolated AddinFlatJy.dll load/export probe.
 *
 * This helper intentionally does not call any exported trading function.  It
 * loads the DLL, enumerates export names from the PE table, reports imported
 * DLL dependencies, and can call the single object factory Addin_GetObject for
 * read-only object layout inspection.  The object probe never calls vtable
 * methods or feature/trading dispatchers.
 *
 * Example compile with MinGW:
 *   gcc -Wall -Wextra -O2 -o addinflatjy_probe.exe addinflatjy_probe.c
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif

#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif

static const char *DEFAULT_DLL = "C:\\silkriver\\TCPlugins\\AddinFlatJy.dll";
static const DWORD DEFAULT_WORDS = 16;

typedef void *(__cdecl *AddinGetObjectFn)(void);
typedef BOOL(WINAPI *SetDllDirectoryAFn)(LPCSTR);

static const char *arg_value(int argc, char **argv, const char *name) {
    int i;
    size_t len = strlen(name);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (strncmp(argv[i], name, len) == 0 && argv[i][len] == '=') {
            return argv[i] + len + 1;
        }
    }
    return NULL;
}

static int has_arg(int argc, char **argv, const char *name) {
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *base_name(const char *path) {
    const char *slash = strrchr(path, '\\');
    const char *fwd = strrchr(path, '/');
    const char *base = slash > fwd ? slash : fwd;
    return base ? base + 1 : path;
}

static void print_last_error(const char *prefix, const char *path) {
    if (path) {
        fprintf(stderr, "ERROR %s err=%lu path=%s\n", prefix, GetLastError(), path);
    } else {
        fprintf(stderr, "ERROR %s err=%lu\n", prefix, GetLastError());
    }
}

static DWORD parse_dword_arg(int argc, char **argv, const char *name, DWORD fallback) {
    const char *value = arg_value(argc, argv, name);
    char *end = NULL;
    unsigned long parsed;
    if (!value || !value[0]) {
        return fallback;
    }
    parsed = strtoul(value, &end, 0);
    if (end == value || *end != 0 || parsed == 0 || parsed > 128) {
        fprintf(stderr, "WARN invalid %s=%s using %lu\n", name, value, fallback);
        return fallback;
    }
    return (DWORD)parsed;
}

static void dirname_of(const char *path, char *out, size_t out_size) {
    const char *slash = strrchr(path, '\\');
    const char *fwd = strrchr(path, '/');
    const char *end = slash > fwd ? slash : fwd;
    size_t len;
    if (!end) {
        out[0] = 0;
        return;
    }
    len = (size_t)(end - path);
    if (len >= out_size) {
        len = out_size - 1;
    }
    memcpy(out, path, len);
    out[len] = 0;
}

static void safe_set_dll_directory(const char *dir) {
    HMODULE kernel32;
    SetDllDirectoryAFn set_dll_directory;
    if (!dir || !dir[0]) {
        return;
    }
    kernel32 = GetModuleHandleA("kernel32.dll");
    if (!kernel32) {
        return;
    }
    set_dll_directory = (SetDllDirectoryAFn)GetProcAddress(kernel32, "SetDllDirectoryA");
    if (set_dll_directory) {
        set_dll_directory(dir);
    }
}

static DWORD rva_to_offset(DWORD rva, IMAGE_SECTION_HEADER *sections, WORD count) {
    WORD i;
    for (i = 0; i < count; i++) {
        DWORD start = sections[i].VirtualAddress;
        DWORD size = sections[i].Misc.VirtualSize;
        if (sections[i].SizeOfRawData > size) {
            size = sections[i].SizeOfRawData;
        }
        if (rva >= start && rva < start + size) {
            return sections[i].PointerToRawData + (rva - start);
        }
    }
    return rva;
}

static int map_pe(const char *dll_path, HANDLE *file, HANDLE *mapping, BYTE **view) {
    *file = CreateFileA(dll_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (*file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "ERROR CreateFile failed err=%lu path=%s\n", GetLastError(), dll_path);
        return 0;
    }
    *mapping = CreateFileMappingA(*file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!*mapping) {
        fprintf(stderr, "ERROR CreateFileMapping failed err=%lu\n", GetLastError());
        CloseHandle(*file);
        return 0;
    }
    *view = (BYTE *)MapViewOfFile(*mapping, FILE_MAP_READ, 0, 0, 0);
    if (!*view) {
        fprintf(stderr, "ERROR MapViewOfFile failed err=%lu\n", GetLastError());
        CloseHandle(*mapping);
        CloseHandle(*file);
        return 0;
    }
    return 1;
}

static IMAGE_NT_HEADERS *nt_headers(BYTE *image) {
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)image;
    IMAGE_NT_HEADERS *nt;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        return NULL;
    }
    nt = (IMAGE_NT_HEADERS *)(image + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) {
        return NULL;
    }
    return nt;
}

static DWORD module_size_from_base(HMODULE mod) {
    BYTE *base = (BYTE *)mod;
    IMAGE_NT_HEADERS *nt = nt_headers(base);
    if (!nt) {
        return 0;
    }
    return nt->OptionalHeader.SizeOfImage;
}

static int addr_in_range(DWORD value, HMODULE mod, DWORD size) {
    DWORD start = (DWORD)(ULONG_PTR)mod;
    DWORD end = start + size;
    return size != 0 && value >= start && value < end;
}

static void format_protect(DWORD protect, char *out, size_t out_size) {
    DWORD base = protect & 0xff;
    const char *name = "UNKNOWN";
    switch (base) {
    case PAGE_EXECUTE:
        name = "X";
        break;
    case PAGE_EXECUTE_READ:
        name = "XR";
        break;
    case PAGE_EXECUTE_READWRITE:
        name = "XRW";
        break;
    case PAGE_EXECUTE_WRITECOPY:
        name = "XWC";
        break;
    case PAGE_READONLY:
        name = "R";
        break;
    case PAGE_READWRITE:
        name = "RW";
        break;
    case PAGE_WRITECOPY:
        name = "WC";
        break;
    case PAGE_NOACCESS:
        name = "NOACCESS";
        break;
    default:
        break;
    }
    snprintf(out, out_size, "%s%s%s",
             name,
             (protect & PAGE_GUARD) ? "|GUARD" : "",
             (protect & PAGE_NOCACHE) ? "|NOCACHE" : "");
}

static int read_memory(const void *addr, void *out, SIZE_T size) {
    SIZE_T got = 0;
    if (!addr) {
        return 0;
    }
    if (!ReadProcessMemory(GetCurrentProcess(), addr, out, size, &got)) {
        return 0;
    }
    return got == size;
}

static int is_probably_readable(DWORD value) {
    MEMORY_BASIC_INFORMATION mbi;
    DWORD protect;
    if (value == 0) {
        return 0;
    }
    if (!VirtualQuery((LPCVOID)(ULONG_PTR)value, &mbi, sizeof(mbi))) {
        return 0;
    }
    if (mbi.State != MEM_COMMIT) {
        return 0;
    }
    protect = mbi.Protect & 0xff;
    if (mbi.Protect & PAGE_GUARD) {
        return 0;
    }
    return protect == PAGE_READONLY || protect == PAGE_READWRITE ||
           protect == PAGE_WRITECOPY || protect == PAGE_EXECUTE_READ ||
           protect == PAGE_EXECUTE_READWRITE || protect == PAGE_EXECUTE_WRITECOPY;
}

static int is_probably_executable(DWORD value) {
    MEMORY_BASIC_INFORMATION mbi;
    DWORD protect;
    if (value == 0) {
        return 0;
    }
    if (!VirtualQuery((LPCVOID)(ULONG_PTR)value, &mbi, sizeof(mbi))) {
        return 0;
    }
    if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) {
        return 0;
    }
    protect = mbi.Protect & 0xff;
    return protect == PAGE_EXECUTE || protect == PAGE_EXECUTE_READ ||
           protect == PAGE_EXECUTE_READWRITE || protect == PAGE_EXECUTE_WRITECOPY;
}

static void print_address_judgement(const char *prefix, DWORD value, HMODULE target_mod, DWORD target_size) {
    MEMORY_BASIC_INFORMATION mbi;
    HMODULE owner = NULL;
    char owner_path[MAX_PATH * 2];
    char protect[64];
    DWORD owner_size = 0;
    const char *owner_name = "<none>";

    owner_path[0] = 0;
    protect[0] = 0;
    if (value != 0 && VirtualQuery((LPCVOID)(ULONG_PTR)value, &mbi, sizeof(mbi))) {
        format_protect(mbi.Protect, protect, sizeof(protect));
        if (mbi.Type == MEM_IMAGE && mbi.AllocationBase) {
            owner = (HMODULE)mbi.AllocationBase;
            owner_size = module_size_from_base(owner);
            if (GetModuleFileNameA(owner, owner_path, sizeof(owner_path))) {
                owner_name = base_name(owner_path);
            }
        }
        printf("%s addr=0x%08lx in_addin=%d readable=%d executable=%d region_base=%p region_size=0x%lx protect=%s owner=%p owner_size=0x%lx owner_name=%s owner_offset=0x%08lx\n",
               prefix,
               value,
               addr_in_range(value, target_mod, target_size),
               is_probably_readable(value),
               is_probably_executable(value),
               mbi.BaseAddress,
               (unsigned long)mbi.RegionSize,
               protect,
               (void *)owner,
               owner_size,
               owner_name,
               owner ? value - (DWORD)(ULONG_PTR)owner : 0UL);
        return;
    }
    printf("%s addr=0x%08lx in_addin=%d readable=0 executable=0 region_base=<none> region_size=0x0 protect=<none> owner=<none> owner_size=0x0 owner_name=<none> owner_offset=0x00000000\n",
           prefix, value, addr_in_range(value, target_mod, target_size));
}

static int print_exports(const char *dll_path) {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = NULL;
    BYTE *image = NULL;
    IMAGE_NT_HEADERS *nt;
    IMAGE_SECTION_HEADER *sections;
    IMAGE_DATA_DIRECTORY dir;
    IMAGE_EXPORT_DIRECTORY *exports;
    DWORD names_offset;
    DWORD *names;
    DWORD i;

    if (!map_pe(dll_path, &file, &mapping, &image)) {
        return 1;
    }
    nt = nt_headers(image);
    if (!nt) {
        fprintf(stderr, "ERROR invalid PE image\n");
        return 1;
    }
    sections = IMAGE_FIRST_SECTION(nt);
    dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    printf("OK PE machine=0x%04x sections=%u export_rva=0x%lx export_size=%lu\n",
           nt->FileHeader.Machine, nt->FileHeader.NumberOfSections, dir.VirtualAddress, dir.Size);
    if (dir.VirtualAddress == 0) {
        printf("OK FUNCTIONS count=0\n");
        return 0;
    }
    exports = (IMAGE_EXPORT_DIRECTORY *)(image + rva_to_offset(
        dir.VirtualAddress, sections, nt->FileHeader.NumberOfSections));
    names_offset = rva_to_offset(exports->AddressOfNames, sections, nt->FileHeader.NumberOfSections);
    names = (DWORD *)(image + names_offset);
    printf("OK FUNCTIONS count=%lu\n", exports->NumberOfNames);
    for (i = 0; i < exports->NumberOfNames; i++) {
        const char *name = (const char *)(image + rva_to_offset(names[i], sections, nt->FileHeader.NumberOfSections));
        printf("EXPORT %s\n", name);
    }

    UnmapViewOfFile(image);
    CloseHandle(mapping);
    CloseHandle(file);
    return 0;
}

static int print_dependencies(const char *dll_path) {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = NULL;
    BYTE *image = NULL;
    IMAGE_NT_HEADERS *nt;
    IMAGE_SECTION_HEADER *sections;
    IMAGE_DATA_DIRECTORY dir;
    IMAGE_IMPORT_DESCRIPTOR *imports;
    int count = 0;

    if (!map_pe(dll_path, &file, &mapping, &image)) {
        return 1;
    }
    nt = nt_headers(image);
    if (!nt) {
        fprintf(stderr, "ERROR invalid PE image\n");
        return 1;
    }
    sections = IMAGE_FIRST_SECTION(nt);
    dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dir.VirtualAddress == 0) {
        printf("OK DEPENDENCIES count=0\n");
        return 0;
    }
    imports = (IMAGE_IMPORT_DESCRIPTOR *)(image + rva_to_offset(
        dir.VirtualAddress, sections, nt->FileHeader.NumberOfSections));
    while (imports[count].Name) {
        const char *name = (const char *)(image + rva_to_offset(imports[count].Name, sections, nt->FileHeader.NumberOfSections));
        printf("DEPENDENCY %s\n", name);
        count++;
    }
    printf("OK DEPENDENCIES count=%d\n", count);

    UnmapViewOfFile(image);
    CloseHandle(mapping);
    CloseHandle(file);
    return 0;
}

static int load_only(const char *dll_path) {
    HMODULE mod;
    char dll_dir[MAX_PATH * 2];
    dirname_of(dll_path, dll_dir, sizeof(dll_dir));
    if (dll_dir[0]) {
        safe_set_dll_directory(dll_dir);
    }
    mod = LoadLibraryExA(dll_path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!mod) {
        mod = LoadLibraryA(dll_path);
    }
    if (!mod) {
        fprintf(stderr, "ERROR LOAD failed err=%lu path=%s\n", GetLastError(), dll_path);
        return 1;
    }
    printf("OK LOAD module=%p dll=%s no_exports_called=1\n", (void *)mod, base_name(dll_path));
    FreeLibrary(mod);
    return 0;
}

static HMODULE load_addin_module(const char *dll_path) {
    HMODULE mod;
    char dll_dir[MAX_PATH * 2];
    char cwd[MAX_PATH * 2];

    dirname_of(dll_path, dll_dir, sizeof(dll_dir));
    if (dll_dir[0]) {
        safe_set_dll_directory(dll_dir);
        (void)cwd;
    }
    mod = LoadLibraryExA(dll_path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!mod) {
        mod = LoadLibraryA(dll_path);
    }
    return mod;
}

static int object_probe(const char *dll_path, DWORD words) {
    HMODULE mod;
    DWORD module_size;
    FARPROC proc;
    AddinGetObjectFn get_object;
    void *object_ptr;
    DWORD object_words[128];
    DWORD vtable_words[128];
    DWORD vtable_ptr;
    DWORD i;
    int object_read_ok;
    int vtable_read_ok;

    if (sizeof(void *) != 4) {
        fprintf(stderr, "ERROR object probe must run in a 32-bit process pointer_size=%lu\n",
                (unsigned long)sizeof(void *));
        return 1;
    }

    mod = load_addin_module(dll_path);
    if (!mod) {
        print_last_error("LOAD failed", dll_path);
        return 1;
    }

    module_size = module_size_from_base(mod);
    printf("OK LOAD module=%p module_size=0x%lx module_end=%p dll=%s\n",
           (void *)mod,
           module_size,
           (void *)((BYTE *)mod + module_size),
           base_name(dll_path));
    print_address_judgement("MODULE_RANGE", (DWORD)(ULONG_PTR)mod, mod, module_size);

    proc = GetProcAddress(mod, "Addin_GetObject");
    if (!proc) {
        print_last_error("GetProcAddress Addin_GetObject failed", NULL);
        FreeLibrary(mod);
        return 1;
    }
    printf("OK EXPORT Addin_GetObject=%p in_addin=%d safe_factory_call_only=1\n",
           (void *)proc,
           addr_in_range((DWORD)(ULONG_PTR)proc, mod, module_size));
    print_address_judgement("EXPORT_ADDR", (DWORD)(ULONG_PTR)proc, mod, module_size);

    get_object = (AddinGetObjectFn)proc;
    object_ptr = get_object();
    printf("OBJECT_PTR addr=%p null=%d\n", object_ptr, object_ptr == NULL);
    print_address_judgement("OBJECT_ADDR", (DWORD)(ULONG_PTR)object_ptr, mod, module_size);
    if (!object_ptr) {
        printf("OK OBJECT_PROBE no_vtable_calls=1 object_null=1\n");
        FreeLibrary(mod);
        return 0;
    }

    memset(object_words, 0, sizeof(object_words));
    object_read_ok = read_memory(object_ptr, object_words, words * sizeof(DWORD));
    printf("OBJECT_DWORDS count=%lu read_ok=%d\n", words, object_read_ok);
    if (!object_read_ok) {
        printf("OK OBJECT_PROBE no_vtable_calls=1 object_read_ok=0\n");
        FreeLibrary(mod);
        return 0;
    }
    for (i = 0; i < words; i++) {
        char label[64];
        snprintf(label, sizeof(label), "OBJECT_DWORD[%lu]", (unsigned long)i);
        printf("%s value=0x%08lx\n", label, object_words[i]);
        print_address_judgement(label, object_words[i], mod, module_size);
    }

    vtable_ptr = object_words[0];
    printf("VTABLE_CANDIDATE from_object_dword=0 addr=0x%08lx readable=%d\n",
           vtable_ptr,
           is_probably_readable(vtable_ptr));
    print_address_judgement("VTABLE_ADDR", vtable_ptr, mod, module_size);
    if (!is_probably_readable(vtable_ptr)) {
        printf("OK OBJECT_PROBE no_vtable_calls=1 vtable_read_ok=0\n");
        FreeLibrary(mod);
        return 0;
    }

    memset(vtable_words, 0, sizeof(vtable_words));
    vtable_read_ok = read_memory((const void *)(ULONG_PTR)vtable_ptr, vtable_words, words * sizeof(DWORD));
    printf("VTABLE_DWORDS count=%lu read_ok=%d no_calls=1\n", words, vtable_read_ok);
    if (vtable_read_ok) {
        for (i = 0; i < words; i++) {
            char label[64];
            snprintf(label, sizeof(label), "VTABLE_FN[%lu]", (unsigned long)i);
            printf("%s addr=0x%08lx\n", label, vtable_words[i]);
            print_address_judgement(label, vtable_words[i], mod, module_size);
        }
    }

    printf("OK OBJECT_PROBE no_vtable_calls=1 no_feature_calls=1 no_buy_sell_calls=1\n");
    FreeLibrary(mod);
    return 0;
}

int main(int argc, char **argv) {
    const char *dll_path = arg_value(argc, argv, "--dll");
    DWORD words = parse_dword_arg(argc, argv, "--words", DEFAULT_WORDS);
    if (!dll_path) {
        dll_path = arg_value(argc, argv, "--dll-path");
    }
    if (!dll_path) {
        dll_path = DEFAULT_DLL;
    }

    if (has_arg(argc, argv, "--help")) {
        printf("Usage: addinflatjy_probe.exe [--dll PATH] [--words N] [--functions|--dependencies|--load|--object]\n");
        printf("Safety: --object calls Addin_GetObject only, then reads object/vtable memory; it never calls vtable, feature, BUY, or SELL methods.\n");
        return 0;
    }
    if (has_arg(argc, argv, "--dependencies")) {
        return print_dependencies(dll_path);
    }
    if (has_arg(argc, argv, "--load")) {
        return load_only(dll_path);
    }
    if (has_arg(argc, argv, "--object")) {
        return object_probe(dll_path, words);
    }
    return print_exports(dll_path);
}

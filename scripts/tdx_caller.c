/*
 * tdx_caller.c - 32-bit helper to call Tc.dll trading functions.
 * Communicates with parent Python process via stdin/stdout.
 *
 * Compile: gcc -m32 -o tdx_caller.exe tdx_caller.c -L. -lTc
 * Or:     gcc -m32 -o tdx_caller.exe tdx_caller.c
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifndef SetDllDirectoryA
WINBASEAPI BOOL WINAPI SetDllDirectoryA(LPCSTR lpPathName);
#endif

typedef int  (*TC_GetVersion_t)(float*, char*);
typedef int  (*TC_GetJyStatus_t)(char*, int*, char*);
typedef int  (*TC_Login_t)(const char*, const char*, const char*);
typedef int  (*TC_Login2_t)(const char*, const char*, const char*, const char*);
typedef void* (*TC_GetLoginRet_t)(void);
typedef int  (*TC_CreateAll_t)(unsigned int, unsigned int);
typedef int  (*TC_Init_Environ_t)(const char*);
typedef int  (*TC_DoGridJy_t)(const char*, const char*, const char*, const char*, const char*, const char*, const char*);
typedef int  (*TC_DoLevinJy_t)(const char*);
typedef void* (*TC_GetClientInfo_t)(void);
typedef int  (*TC_OperateUser_t)(const char*);
typedef void (*TC_Uninit_t)(void);

static char g_current_call[128] = "startup";

static void diag(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
}

static void set_current_call(const char *name) {
    strncpy(g_current_call, name, sizeof(g_current_call) - 1);
    g_current_call[sizeof(g_current_call) - 1] = 0;
}

static LONG WINAPI crash_filter(EXCEPTION_POINTERS *info) {
    DWORD code = 0;
    void *address = NULL;
    if (info && info->ExceptionRecord) {
        code = info->ExceptionRecord->ExceptionCode;
        address = info->ExceptionRecord->ExceptionAddress;
    }
    diag("CRASH function=%s exception=0x%08lX address=%p", g_current_call, code, address);
    return EXCEPTION_EXECUTE_HANDLER;
}

static int has_arg(int argc, char *argv[], const char *needle) {
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], needle) == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *arg_value(int argc, char *argv[], const char *name) {
    int i;
    size_t prefix_len = strlen(name);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (strncmp(argv[i], name, prefix_len) == 0 && argv[i][prefix_len] == '=') {
            return argv[i] + prefix_len + 1;
        }
    }
    return NULL;
}

static int live_enabled(int argc, char *argv[]) {
    const char *env = getenv("TDX_CALLER_ALLOW_LIVE");
    return has_arg(argc, argv, "--allow-live") || (env && strcmp(env, "1") == 0);
}

static HMODULE load_tc_dll(int argc, char *argv[]) {
    const char *work_dir = arg_value(argc, argv, "--work-dir");
    const char *dll_dir = arg_value(argc, argv, "--dll-dir");
    char dll_path[MAX_PATH * 2];

    if (work_dir && !SetCurrentDirectoryA(work_dir)) {
        diag("WARN SetCurrentDirectoryA failed dir=%s err=%lu", work_dir, GetLastError());
    }
    if (dll_dir && !SetDllDirectoryA(dll_dir)) {
        diag("WARN SetDllDirectoryA failed dir=%s err=%lu", dll_dir, GetLastError());
    }
    if (dll_dir && strlen(dll_dir) + strlen("\\Tc.dll") < sizeof(dll_path)) {
        snprintf(dll_path, sizeof(dll_path), "%s\\Tc.dll", dll_dir);
        diag("Loading Tc.dll path=%s work_dir=%s", dll_path, work_dir ? work_dir : "");
        return LoadLibraryA(dll_path);
    }
    diag("Loading Tc.dll via default DLL search work_dir=%s", work_dir ? work_dir : "");
    return LoadLibraryA("Tc.dll");
}

static void print_functions(
    TC_GetVersion_t pGetVersion,
    TC_GetJyStatus_t pGetJyStatus,
    TC_Login_t pLogin,
    TC_Login2_t pLogin2,
    TC_GetLoginRet_t pGetLoginRet,
    TC_CreateAll_t pCreateAll,
    TC_Init_Environ_t pInitEnviron,
    TC_DoGridJy_t pDoGridJy,
    TC_DoLevinJy_t pDoLevinJy,
    TC_GetClientInfo_t pGetClientInfo,
    TC_OperateUser_t pOperateUser,
    TC_Uninit_t pUninit
) {
    fprintf(stdout,
            "OK FUNCTIONS TC_GetVersion=%d TC_GetJyStatus=%d TC_Login=%d TC_Login2=%d "
            "TC_GetLoginRet=%d TC_CreateAll=%d TC_Init_Environ=%d TC_DoGridJy=%d "
            "TC_DoLevinJy=%d TC_GetClientInfo=%d TC_OperateUser=%d TC_Uninit=%d\n",
            pGetVersion != NULL, pGetJyStatus != NULL, pLogin != NULL, pLogin2 != NULL,
            pGetLoginRet != NULL, pCreateAll != NULL, pInitEnviron != NULL,
            pDoGridJy != NULL, pDoLevinJy != NULL, pGetClientInfo != NULL,
            pOperateUser != NULL, pUninit != NULL);
}

int main(int argc, char *argv[]) {
    int allow_live = live_enabled(argc, argv);
    HMODULE hMod;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    SetUnhandledExceptionFilter(crash_filter);

    hMod = load_tc_dll(argc, argv);
    if (!hMod) {
        fprintf(stderr, "ERROR: Cannot load Tc.dll (err=%lu)\n", GetLastError());
        return 1;
    }
    diag("Tc.dll loaded at %p live_enabled=%d", hMod, allow_live);

    /* Resolve functions */
    TC_GetVersion_t     pGetVersion    = (TC_GetVersion_t)    GetProcAddress(hMod, "TC_GetVersion");
    TC_GetJyStatus_t    pGetJyStatus   = (TC_GetJyStatus_t)   GetProcAddress(hMod, "TC_GetJyStatus");
    TC_GetLoginRet_t    pGetLoginRet   = (TC_GetLoginRet_t)   GetProcAddress(hMod, "TC_GetLoginRet");
    TC_CreateAll_t      pCreateAll     = (TC_CreateAll_t)     GetProcAddress(hMod, "TC_CreateAll");
    TC_Login_t          pLogin         = (TC_Login_t)         GetProcAddress(hMod, "TC_Login");
    TC_Login2_t         pLogin2        = (TC_Login2_t)        GetProcAddress(hMod, "TC_Login2");
    TC_Init_Environ_t   pInitEnviron   = (TC_Init_Environ_t)  GetProcAddress(hMod, "TC_Init_Environ");
    TC_DoGridJy_t       pDoGridJy      = (TC_DoGridJy_t)      GetProcAddress(hMod, "TC_DoGridJy");
    TC_DoLevinJy_t      pDoLevinJy     = (TC_DoLevinJy_t)     GetProcAddress(hMod, "TC_DoLevinJy");
    TC_GetClientInfo_t  pGetClientInfo = (TC_GetClientInfo_t) GetProcAddress(hMod, "TC_GetClientInfo");
    TC_OperateUser_t    pOperateUser   = (TC_OperateUser_t)   GetProcAddress(hMod, "TC_OperateUser");
    TC_Uninit_t         pUninit        = (TC_Uninit_t)        GetProcAddress(hMod, "TC_Uninit");

    diag("Functions resolved:");
    diag("  TC_GetVersion=%p TC_GetJyStatus=%p TC_Login=%p", pGetVersion, pGetJyStatus, pLogin);
    diag("  TC_CreateAll=%p TC_DoGridJy=%p TC_GetLoginRet=%p", pCreateAll, pDoGridJy, pGetLoginRet);

    /* Interactive command loop */
    char line[4096];
    fprintf(stdout, "READY\n");

    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline */
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char *cr = strchr(line, '\r');
        if (cr) *cr = 0;

        set_current_call("idle");

        if (strcmp(line, "QUIT") == 0) {
            break;
        }
        else if (strcmp(line, "PING") == 0) {
            fprintf(stdout, "OK PING\n");
        }
        else if (strcmp(line, "FUNCTIONS") == 0) {
            print_functions(
                pGetVersion, pGetJyStatus, pLogin, pLogin2, pGetLoginRet, pCreateAll,
                pInitEnviron, pDoGridJy, pDoLevinJy, pGetClientInfo, pOperateUser, pUninit);
        }
        else if (strcmp(line, "VERSION") == 0) {
            if (pGetVersion) {
                float ver = 0.0f;
                char text[256] = {0};
                set_current_call("TC_GetVersion");
                diag("CALL_BEGIN function=TC_GetVersion command=VERSION args=float*,char[256]");
                int ret = pGetVersion(&ver, text);
                diag("CALL_END function=TC_GetVersion ret=%d version_float=%.6g version_text=%s", ret, ver, text);
                set_current_call("idle");
                fprintf(stdout, "OK VERSION ret=%d float=%.6g text=%s\n", ret, ver, text);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strcmp(line, "STATUS") == 0) {
            diag("BLOCKED function=TC_GetJyStatus reason=unsafe_internal_state_use STATUS_RAW_required=1");
            fprintf(stdout, "ERR unsafe STATUS disabled; use STATUS_RAW for crash diagnostics\n");
        }
        else if (strcmp(line, "STATUS_RAW") == 0) {
            if (pGetJyStatus) {
                char status_text[256] = {0};
                int status_code = 0;
                char detail_text[256] = {0};
                set_current_call("TC_GetJyStatus");
                diag("CALL_BEGIN function=TC_GetJyStatus command=STATUS_RAW args=char[256],int*,char[256]");
                int st = pGetJyStatus(status_text, &status_code, detail_text);
                diag("CALL_END function=TC_GetJyStatus ret=%d status_code=%d status=%s detail=%s",
                     st, status_code, status_text, detail_text);
                set_current_call("idle");
                fprintf(stdout, "OK STATUS_RAW ret=%d code=%d status=%s detail=%s\n",
                        st, status_code, status_text, detail_text);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strncmp(line, "LOGIN ", 6) == 0) {
            /* LOGIN account password extra */
            char account[256] = {0}, password[256] = {0}, extra[256] = {0};
            int n = sscanf(line + 6, "%255s %255s %255s", account, password, extra);
            if (n >= 2 && pLogin) {
                set_current_call("TC_Login");
                diag("CALL_BEGIN function=TC_Login account=%s extra=%s", account, n >= 3 ? extra : "");
                int ret = pLogin(account, password, n >= 3 ? extra : "");
                diag("CALL_END function=TC_Login ret=%d", ret);
                set_current_call("idle");
                fprintf(stdout, "OK LOGIN=%d\n", ret);
            } else {
                fprintf(stdout, "ERR usage: LOGIN account password [extra]\n");
            }
        }
        else if (strncmp(line, "GRIDJY ", 7) == 0) {
            /* GRIDJY entry params */
            char entry[256] = {0}, params[2048] = {0};
            char *space = strchr(line + 7, ' ');
            if (space) {
                *space = 0;
                strncpy(entry, line + 7, 255);
                strncpy(params, space + 1, 2047);
            } else {
                strncpy(entry, line + 7, 255);
            }
            diag("BLOCKED function=TC_DoGridJy reason=unsafe_signature_unconfirmed entry=%s live_enabled=%d",
                 entry, allow_live);
            fprintf(stdout, "ERR unsafe GRIDJY disabled; use GRIDJY_DRYRUN for construction checks\n");
        }
        else if (strncmp(line, "GRIDJY_DRYRUN ", 14) == 0) {
            char entry[256] = {0}, params[2048] = {0};
            char *space = strchr(line + 14, ' ');
            if (space) {
                *space = 0;
                strncpy(entry, line + 14, 255);
                strncpy(params, space + 1, 2047);
            } else {
                strncpy(entry, line + 14, 255);
            }
            diag("DRYRUN function=TC_DoGridJy entry=%s params=%s args=7 no_dll_call=1", entry, params);
            fprintf(stdout, "OK GRIDJY_DRYRUN entry=%s params_len=%u args=7 no_dll_call=1\n",
                    entry, (unsigned int)strlen(params));
        }
        else if (strncmp(line, "LEVINJY ", 8) == 0) {
            if (!allow_live) {
                diag("BLOCKED function=TC_DoLevinJy reason=live_disabled");
                fprintf(stdout, "ERR live disabled for LEVINJY\n");
            } else if (pDoLevinJy) {
                set_current_call("TC_DoLevinJy");
                diag("CALL_BEGIN function=TC_DoLevinJy");
                int ret = pDoLevinJy(line + 8);
                diag("CALL_END function=TC_DoLevinJy ret=%d", ret);
                set_current_call("idle");
                fprintf(stdout, "OK LEVINJY=%d\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strncmp(line, "OPERATE ", 8) == 0) {
            if (!allow_live) {
                diag("BLOCKED function=TC_OperateUser reason=live_disabled");
                fprintf(stdout, "ERR live disabled for OPERATE\n");
            } else if (pOperateUser) {
                set_current_call("TC_OperateUser");
                diag("CALL_BEGIN function=TC_OperateUser");
                int ret = pOperateUser(line + 8);
                diag("CALL_END function=TC_OperateUser ret=%d", ret);
                set_current_call("idle");
                fprintf(stdout, "OK OPERATE=%d\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strcmp(line, "CREATEALL") == 0) {
            if (pCreateAll) {
                set_current_call("TC_CreateAll");
                diag("CALL_BEGIN function=TC_CreateAll command=CREATEALL args=0,0");
                int ret = pCreateAll(0, 0);
                diag("CALL_END function=TC_CreateAll ret=%d", ret);
                set_current_call("idle");
                fprintf(stdout, "OK CREATEALL=%d\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strncmp(line, "INITENV ", 8) == 0) {
            if (pInitEnviron) {
                set_current_call("TC_Init_Environ");
                diag("CALL_BEGIN function=TC_Init_Environ");
                int ret = pInitEnviron(line + 8);
                diag("CALL_END function=TC_Init_Environ ret=%d", ret);
                set_current_call("idle");
                fprintf(stdout, "OK INITENV=%d\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strcmp(line, "GETLOGINRET") == 0) {
            if (pGetLoginRet) {
                set_current_call("TC_GetLoginRet");
                diag("CALL_BEGIN function=TC_GetLoginRet command=GETLOGINRET");
                void *ret = pGetLoginRet();
                diag("CALL_END function=TC_GetLoginRet ret=%p", ret);
                set_current_call("idle");
                fprintf(stdout, "OK GETLOGINRET=%p\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else if (strcmp(line, "CLIENTINFO") == 0) {
            if (pGetClientInfo) {
                set_current_call("TC_GetClientInfo");
                diag("CALL_BEGIN function=TC_GetClientInfo command=CLIENTINFO");
                void *ret = pGetClientInfo();
                diag("CALL_END function=TC_GetClientInfo ret=%p", ret);
                set_current_call("idle");
                fprintf(stdout, "OK CLIENTINFO=%p\n", ret);
            } else {
                fprintf(stdout, "ERR function not found\n");
            }
        }
        else {
            fprintf(stdout, "ERR unknown command: %s\n", line);
        }
    }

    set_current_call("FreeLibrary");
    if (pUninit) {
        set_current_call("TC_Uninit");
        diag("CALL_BEGIN function=TC_Uninit");
        pUninit();
        diag("CALL_END function=TC_Uninit");
    }
    FreeLibrary(hMod);
    return 0;
}

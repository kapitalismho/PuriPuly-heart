#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <shellapi.h>
#include <windows.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "flutter_window.h"
#include "native_layout.generated.h"
#include "utils.h"

namespace {
using SpArgvFn = int (*)(int, wchar_t **);

std::filesystem::path ExecutableRoot() {
  std::wstring buffer(32768, L'\0');
  const DWORD length = ::GetModuleFileNameW(nullptr, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
  if (length == 0 || length == buffer.size()) {
    return {};
  }
  buffer.resize(length);
  return std::filesystem::path(buffer).parent_path();
}

std::wstring ProductArgvJson(int argc, wchar_t **argv) {
  constexpr wchar_t kHex[] = L"0123456789abcdef";
  std::wstring result = L"[\"PuriPulyHeart\"";
  for (int index = 1; index < argc; ++index) {
    result.append(L",\"");
    for (const wchar_t value : std::wstring_view(argv[index])) {
      result.append(L"\\u");
      result.push_back(kHex[(value >> 12) & 0xF]);
      result.push_back(kHex[(value >> 8) & 0xF]);
      result.push_back(kHex[(value >> 4) & 0xF]);
      result.push_back(kHex[value & 0xF]);
    }
    result.push_back(L'"');
  }
  result.push_back(L']');
  return result;
}

bool SetProductArguments(int argc, wchar_t **argv) {
  const auto payload = ProductArgvJson(argc, argv);
  if (!::SetEnvironmentVariableW(L"PURIPULY_HEART_NATIVE_ARGV_JSON",
                                 payload.c_str())) {
    return false;
  }
  return ::_wputenv_s(L"PURIPULY_HEART_NATIVE_ARGV_JSON", payload.c_str()) == 0;
}

void SetProcessEnvironment(const wchar_t *name, const std::wstring &value) {
  ::SetEnvironmentVariableW(name, value.c_str());
  ::_wputenv_s(name, value.c_str());
}

void ClearProcessEnvironment(const wchar_t *name) {
  ::SetEnvironmentVariableW(name, nullptr);
  ::_wputenv_s(name, L"");
}

bool ConfigureInstalledEnvironment(const std::filesystem::path &root) {
  if (root.empty()) {
    return false;
  }
  const auto app = root / puripuly_layout::kApplicationRoot;
  const auto dependencies = root / puripuly_layout::kDependencyRoot;
  const auto dlls = root / puripuly_layout::kExtensionDllRoot;
  const auto host = root / puripuly_layout::kHostExecutable;
  const auto python = root / puripuly_layout::kPythonExecutable;
  if (!std::filesystem::is_regular_file(host) ||
      !std::filesystem::is_regular_file(python) ||
      !std::filesystem::is_directory(app) ||
      !std::filesystem::is_directory(dependencies) ||
      !std::filesystem::is_directory(dlls)) {
    return false;
  }

  wchar_t system_directory[MAX_PATH] = {};
  if (::GetSystemDirectoryW(system_directory, MAX_PATH) == 0) {
    return false;
  }
  const std::wstring path = root.wstring() + L";" + dlls.wstring() + L";" +
                            dependencies.wstring() + L";" + system_directory;
  const std::wstring python_path =
      app.wstring() + L";" + dependencies.wstring();

  SetProcessEnvironment(L"PYTHONHOME", root.wstring());
  SetProcessEnvironment(L"PYTHONPATH", python_path);
  SetProcessEnvironment(L"PATH", path);
  SetProcessEnvironment(L"PYTHONNOUSERSITE", L"1");
  SetProcessEnvironment(L"PYTHONSAFEPATH", L"1");
  SetProcessEnvironment(L"PYTHONDONTWRITEBYTECODE", L"1");
  SetProcessEnvironment(L"PYTHONOPTIMIZE", L"0");
  SetProcessEnvironment(L"PURIPULY_HEART_NATIVE_RESOURCE_ROOT", app.wstring());
  SetProcessEnvironment(L"PURIPULY_HEART_NATIVE_RUNTIME_ROOT", root.wstring());
  SetProcessEnvironment(L"PURIPULY_HEART_NATIVE_HOST_EXECUTABLE",
                        host.wstring());
  SetProcessEnvironment(L"PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE",
                        python.wstring());
  SetProcessEnvironment(L"FLET_HIDE_WINDOW_ON_START", L"1");
  ClearProcessEnvironment(L"FLET_DART_BRIDGE_PORT");
  ClearProcessEnvironment(L"FLET_DART_BRIDGE_EXIT_PORT");

  if (!::SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                  LOAD_LIBRARY_SEARCH_USER_DIRS)) {
    return false;
  }
  if (::AddDllDirectory(root.c_str()) == nullptr ||
      ::AddDllDirectory(dlls.c_str()) == nullptr ||
      ::AddDllDirectory(dependencies.c_str()) == nullptr) {
    return false;
  }
  return true;
}

HMODULE LoadBridge(const std::filesystem::path &root) {
  const auto bridge_path = root / L"dart_bridge.dll";
  return ::LoadLibraryExW(bridge_path.c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                              LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                              LOAD_LIBRARY_SEARCH_USER_DIRS);
}

bool MaybeRunPython(int argc, wchar_t **argv, HMODULE bridge, int &exit_code) {
  if (bridge == nullptr) {
    exit_code = EXIT_FAILURE;
    return true;
  }
  const auto is_mp = reinterpret_cast<SpArgvFn>(
      ::GetProcAddress(bridge, "serious_python_is_mp_invocation_w"));
  const auto run_main = reinterpret_cast<SpArgvFn>(
      ::GetProcAddress(bridge, "serious_python_main_w"));
  if (is_mp == nullptr || run_main == nullptr) {
    exit_code = EXIT_FAILURE;
    return true;
  }
  if (is_mp(argc, argv) != 0) {
    exit_code = run_main(argc, argv);
    return true;
  }
  if (argc < 2 || std::wstring_view(argv[1]) != L"--headless") {
    return false;
  }

  std::vector<std::wstring> values;
  values.emplace_back(argv[0]);
  values.emplace_back(L"-m");
  values.emplace_back(L"puripuly_heart.main");
  for (int index = 2; index < argc; ++index) {
    values.emplace_back(argv[index]);
  }
  std::vector<wchar_t *> python_argv;
  python_argv.reserve(values.size());
  for (auto &value : values) {
    python_argv.push_back(value.data());
  }
  exit_code =
      run_main(static_cast<int>(python_argv.size()), python_argv.data());
  return true;
}
} // namespace

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE previous,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  const auto root = ExecutableRoot();
  if (!ConfigureInstalledEnvironment(root)) {
    return EXIT_FAILURE;
  }

  int argc = 0;
  wchar_t **argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
  if (argv == nullptr) {
    return EXIT_FAILURE;
  }
  HMODULE bridge = LoadBridge(root);
  int python_exit_code = 0;
  if (MaybeRunPython(argc, argv, bridge, python_exit_code)) {
    ::LocalFree(argv);
    return python_exit_code;
  }
  if (!SetProductArguments(argc, argv)) {
    ::LocalFree(argv);
    return EXIT_FAILURE;
  }
  ::LocalFree(argv);

  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");
  project.set_dart_entrypoint_arguments(std::vector<std::string>{});

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(1280, 720);
  if (!window.Create(L"PuriPuly <3", origin, size)) {
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG message;
  while (::GetMessage(&message, nullptr, 0, 0)) {
    ::TranslateMessage(&message);
    ::DispatchMessage(&message);
  }

  ::TerminateProcess(::GetCurrentProcess(), 0);
  return EXIT_SUCCESS;
}

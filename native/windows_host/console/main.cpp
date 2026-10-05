#include <windows.h>
#include <shellapi.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

namespace {
std::wstring Quote(std::wstring_view value) {
  std::wstring result = L"\"";
  unsigned backslashes = 0;
  for (const wchar_t character : value) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'"') {
      result.append(backslashes * 2 + 1, L'\\');
      result.push_back(L'"');
      backslashes = 0;
      continue;
    }
    result.append(backslashes, L'\\');
    backslashes = 0;
    result.push_back(character);
  }
  result.append(backslashes * 2, L'\\');
  result.push_back(L'"');
  return result;
}

std::filesystem::path ExecutableRoot() {
  std::wstring filename(32768, L'\0');
  const DWORD count = ::GetModuleFileNameW(nullptr, filename.data(),
                                          static_cast<DWORD>(filename.size()));
  if (count == 0 || count >= filename.size()) {
    return {};
  }
  filename.resize(count);
  return std::filesystem::path(filename).parent_path();
}

bool SetEnvironment(const wchar_t *name, const std::wstring &value) {
  return ::SetEnvironmentVariableW(name, value.c_str()) != 0;
}

bool ConfigureEnvironment(const std::filesystem::path &root) {
  if (root.empty()) {
    return false;
  }
  const auto app = root / L"app";
  const auto python_archive = app / L"python.zip";
  const auto stdlib_archive = root / L"python314.zip";
  const auto dependencies = root / L"site-packages";
  const auto dlls = root / L"DLLs";
  const auto python = root / L"python.exe";
  const auto host = root / L"PuriPulyHeart.exe";
  if (!std::filesystem::is_regular_file(python) ||
      !std::filesystem::is_regular_file(host) ||
      !std::filesystem::is_regular_file(python_archive) ||
      !std::filesystem::is_regular_file(stdlib_archive) ||
      !std::filesystem::is_directory(app) ||
      !std::filesystem::is_directory(dependencies) ||
      !std::filesystem::is_directory(dlls)) {
    return false;
  }
  wchar_t system_directory[MAX_PATH] = {};
  if (::GetSystemDirectoryW(system_directory, MAX_PATH) == 0) {
    return false;
  }
  return SetEnvironment(L"PYTHONHOME", root.wstring()) &&
         SetEnvironment(L"PYTHONPATH", python_archive.wstring() + L";" +
                                           stdlib_archive.wstring() + L";" +
                                           app.wstring() + L";" +
                                           dependencies.wstring()) &&
         SetEnvironment(L"PATH", root.wstring() + L";" + dlls.wstring() +
                                     L";" + dependencies.wstring() + L";" +
                                     system_directory) &&
         SetEnvironment(L"PYTHONNOUSERSITE", L"1") &&
         SetEnvironment(L"PYTHONSAFEPATH", L"1") &&
         SetEnvironment(L"PYTHONDONTWRITEBYTECODE", L"1") &&
         SetEnvironment(L"PYTHONOPTIMIZE", L"0") &&
         SetEnvironment(L"PYTHONIOENCODING", L"utf-8") &&
         SetEnvironment(L"PURIPULY_HEART_NATIVE_RESOURCE_ROOT", app.wstring()) &&
         SetEnvironment(L"PURIPULY_HEART_NATIVE_RUNTIME_ROOT", root.wstring()) &&
         SetEnvironment(L"PURIPULY_HEART_NATIVE_HOST_EXECUTABLE", host.wstring()) &&
         SetEnvironment(L"PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE", python.wstring()) &&
         ::SetEnvironmentVariableW(L"FLET_DART_BRIDGE_PORT", nullptr) &&
         ::SetEnvironmentVariableW(L"FLET_DART_BRIDGE_EXIT_PORT", nullptr);
}

BOOL WINAPI KeepWaitingForChild(DWORD event) {
  return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}
}

int wmain() {
  const auto root = ExecutableRoot();
  if (!ConfigureEnvironment(root)) {
    fwprintf(stderr, L"puripuly: installed runtime layout is incomplete\n");
    return EXIT_FAILURE;
  }
  int count = 0;
  wchar_t **arguments = ::CommandLineToArgvW(::GetCommandLineW(), &count);
  if (arguments == nullptr) {
    fwprintf(stderr, L"puripuly: cannot parse command line\n");
    return EXIT_FAILURE;
  }
  const auto python = root / L"python.exe";
  std::wstring command = Quote(python.wstring()) + L" -m product_bootstrap cli";
  for (int index = 1; index < count; ++index) {
    command.push_back(L' ');
    command.append(Quote(arguments[index]));
  }
  ::LocalFree(arguments);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION child{};
  if (!::CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, TRUE,
                        0, nullptr, root.c_str(), &startup, &child)) {
    fwprintf(stderr, L"puripuly: cannot start application (%lu)\n",
             ::GetLastError());
    return EXIT_FAILURE;
  }
  ::CloseHandle(child.hThread);
  ::SetConsoleCtrlHandler(KeepWaitingForChild, TRUE);
  const DWORD waited = ::WaitForSingleObject(child.hProcess, INFINITE);
  DWORD exit_code = EXIT_FAILURE;
  if (waited != WAIT_OBJECT_0 || !::GetExitCodeProcess(child.hProcess, &exit_code)) {
    fwprintf(stderr, L"puripuly: cannot read application exit status (%lu)\n",
             ::GetLastError());
    exit_code = EXIT_FAILURE;
  }
  ::CloseHandle(child.hProcess);
  return static_cast<int>(exit_code);
}

// Accept keyboard input in the terminal to manually control state.
// This allows you to quickly prototype a scene's various state inputs without re-rendering each time.
//
// The syntax is as so:
// my_state_value:{t} 1 +
// Commands must strictly start with a state value name, a colon, and then a valid state equation.

#include "StateTester.h"
#include <iostream>
#include <string>
#include <cstdio>
#include "../../Host_Device_Shared/vec.h"
#include "../Pixels.h"
#include "../../IO/IoHelpers.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

#ifdef _WIN32
struct FfplayPreview {
    FILE* stream;
    HANDLE process;
};

FfplayPreview open_ffplay_preview(const string& command) {
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE child_stdin = nullptr;
    HANDLE parent_write = nullptr;
    if (!CreatePipe(&child_stdin, &parent_write, &security, 0) ||
        !SetHandleInformation(parent_write, HANDLE_FLAG_INHERIT, 0)) {
        throw runtime_error("Failed to create ffplay input pipe.");
    }

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_stdin;
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    vector<char> mutable_command(command.begin(), command.end());
    mutable_command.push_back('\0');

    BOOL created = CreateProcessA(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE, 0,
        nullptr, nullptr, &startup, &process
    );
    CloseHandle(child_stdin);
    if (!created) {
        CloseHandle(parent_write);
        throw runtime_error("Failed to start ffplay for the state tester UI.");
    }
    CloseHandle(process.hThread);

    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(parent_write), _O_BINARY);
    FILE* stream = fd < 0 ? nullptr : _fdopen(fd, "wb");
    if (!stream) {
        if (fd >= 0) _close(fd); else CloseHandle(parent_write);
        CloseHandle(process.hProcess);
        throw runtime_error("Failed to open ffplay input stream.");
    }
    return {stream, process.hProcess};
}

void close_ffplay_preview(FfplayPreview& preview) {
    fclose(preview.stream);
    WaitForSingleObject(preview.process, INFINITE);
    CloseHandle(preview.process);
}
#endif

void parse_command(const string& command, string& state_value, string& operation)
{
    size_t colon_pos = command.find(':');
    if (colon_pos == string::npos)
    {
        cerr << "Invalid command format. Expected 'state_value:operation'." << endl;
        return;
    }

    state_value = command.substr(0, colon_pos);
    operation = command.substr(colon_pos + 1);
}

void open_ui(Scene& scene) {
    cout << endl << "State Tester UI" << endl;
    cout << "Enter commands in the format: state_value:operation" << endl;
    cout << "Type 'exit' to keep the desired state value and continue rendering the video." << endl;
    cout << "Type 'print_state' to print the entire local state of the scene." << endl;

    ivec2 dimensions = scene.get_width_height();
    string ffplay_cmd_str = "ffplay -autoexit -loglevel error -f rawvideo -pixel_format argb -video_size " + to_string(dimensions.x) + "x" + to_string(dimensions.y) + " -";
    cout << "Running command: " << ffplay_cmd_str << endl;

#ifdef _WIN32
    FfplayPreview preview = open_ffplay_preview(ffplay_cmd_str);
    FILE* pipe = preview.stream;
#else
    FILE* pipe = portable_popen(ffplay_cmd_str.c_str(), "w");
#endif
    if (!pipe) {
        throw runtime_error("Failed to start ffplay for the state tester UI.");
    }

    string input;
    bool skip_render = false;
    while (true)
    {
        if (!skip_render) {
            uint32_t* gpu_ptr = scene.query();
            Pixels pix(dimensions);
            cuda_copy_pixels_to_host(pix.pixels.data(), pix.pixels.size(), gpu_ptr);

            pix.print_to_terminal();

            fwrite(pix.pixels.data(),
                sizeof(int32_t),
                pix.pixels.size(),
                pipe);

            fflush(pipe);
        }
        skip_render = false;

        cout << "> ";
        getline(cin, input);

        if (input == "exit")
            break;
        else if (input == "print_state") {
            scene.manager.print_state();
            skip_render = true; // We don't want a new picture to hog up the screen
        } else {
            string state_value, operation;
            parse_command(input, state_value, operation);

            // Here you would apply the operation to the state_value
            // For demonstration purposes, we'll just print them out
            cout << "State Value: " << state_value << ", Operation: " << operation << endl;

            scene.manager.set(state_value, operation);
        }
    }
#ifdef _WIN32
    close_ffplay_preview(preview);
#else
    portable_pclose(pipe);
#endif
}

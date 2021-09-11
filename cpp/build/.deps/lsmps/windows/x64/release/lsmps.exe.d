{
    files = {
        [[build\.objs\lsmps\windows\x64\release\lsmps.cpp.obj]]
    },
    values = {
        [[C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\link.exe]],
        {
            "-nologo",
            "-dynamicbase",
            "-nxcompat",
            "-machine:x64",
            "/opt:ref",
            "/opt:icf",
            "Gdi32.lib",
            "User32.lib"
        }
    }
}
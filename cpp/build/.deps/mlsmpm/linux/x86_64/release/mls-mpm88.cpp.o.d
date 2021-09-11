{
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++14",
            "-DNDEBUG"
        }
    },
    files = {
        "mls-mpm88.cpp"
    },
    depfiles_gcc = "build/.objs/mlsmpm/linux/x86_64/release/mls-mpm88.cpp.o: mls-mpm88.cpp  taichi.h\
"
}
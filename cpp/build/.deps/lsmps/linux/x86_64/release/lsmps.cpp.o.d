{
    files = {
        "lsmps.cpp"
    },
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-O3",
            "-std=c++14",
            "-isystem",
            "/home/lwruan/.xmake/packages/e/eigen/3.3.9/4741f970379048e09c876cd7735e7776/include",
            "-isystem",
            "/home/lwruan/.xmake/packages/e/eigen/3.3.9/4741f970379048e09c876cd7735e7776/include/eigen3",
            "-DNDEBUG"
        }
    },
    depfiles_gcc = "build/.objs/lsmps/linux/x86_64/release/lsmps.cpp.o: lsmps.cpp taichi.h\
"
}
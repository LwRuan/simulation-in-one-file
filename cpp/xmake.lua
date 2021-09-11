set_xmakever("2.5.6")

add_requires("eigen")

add_rules("mode.debug", "mode.release")
set_languages("cxx14")

target("lsmps")
    set_kind("binary")
    add_files("lsmps.cpp")
    add_packages("eigen", {public=true})
    if is_os("windows") then
        add_links("Gdi32", "User32")
    elseif is_os("linux") then
        add_links("X11", "pthread")
    end

target("mlsmpm")
    set_kind("binary")
    add_files("mls-mpm88.cpp")
    if is_os("windows") then
        add_links("Gdi32", "User32")
    elseif is_os("linux") then
        add_links("X11", "pthread")
    end
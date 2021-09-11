set_xmakever("2.5.6")

add_requires("eigen")

add_rules("mode.debug", "mode.release")
set_languages("cxx14")

target("lsmps")
    set_kind("binary")
    add_files("lsmps.cpp")
    add_packages("eigen", {public=true})
    add_links("Gdi32", "User32")

target("mlsmpm")
    set_kind("binary")
    add_files("mls-mpm88.cpp")
    add_links("Gdi32", "User32")
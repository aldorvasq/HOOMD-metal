// Copyright (c) 2026 Aldo Vasquez-Briceno.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).

/*! \file hoomd_mslc.mm
    \brief Build-time check of preprocessed HOOMD-blue Metal sources.

    Compiles a Metal Shading Language file with the same options the runtime uses (see
    HipMetalRuntime.mm) and reports diagnostics. Exits with a nonzero status when compilation
    fails. Compiling at build time also populates the system Metal shader cache.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
    {
    @autoreleasepool
        {
        if (argc != 2)
            {
            fprintf(stderr, "usage: hoomd_mslc <file.msl>\n");
            return 2;
            }

        std::ifstream file(argv[1]);
        if (!file)
            {
            fprintf(stderr, "hoomd_mslc: cannot read %s\n", argv[1]);
            return 2;
            }
        std::stringstream text;
        text << file.rdbuf();

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            {
            // Building on a machine without a Metal device: skip the check.
            fprintf(stderr, "hoomd_mslc: no Metal device, skipping check of %s\n", argv[1]);
            return 0;
            }

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_2;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;

        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:text.str().c_str()]
                                 options:options
                                   error:&error];
        if (error)
            {
            printf("%s\n", error.localizedDescription.UTF8String);
            }
        return library ? 0 : 1;
        }
    }

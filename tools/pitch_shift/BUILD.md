# Rubber Band bridge

Vendored upstream source is tag `v4.0.0`, commit
`1d95888bec3ae0a17c0c4af791810d5a63f6bc35` from the official
`breakfastquay/rubberband` repository. License text is retained in
`rubberband-src/COPYING`.

The source repository pins this tree as a Git submodule. After cloning run
`git submodule update --init --recursive`. When using a GitHub ZIP instead,
clone `https://github.com/breakfastquay/rubberband.git` into
`tools/pitch_shift/rubberband-src`, then check out the commit above.

The bridge uses the official `RubberBandStretcher` real-time path with
`OptionFormantPreserved`, `OptionPitchHighConsistency`, and the short window.
It selects the official Finer engine below 3x and Faster at 3x and above;
this avoids the measured high-register drift of Finer at large ratios.
Build the upstream project with its documented Meson build, then compile
`rubberband_bridge.cpp` against the generated library and headers:

The reproducible build used here was:

```text
python -m pip install --target %TEMP%/rb-build-tools meson==1.7.2 ninja==1.11.1.3
python -c "import sys;sys.path.insert(0, r'%TEMP%/rb-build-tools');from mesonbuild.mesonmain import main;sys.argv=['meson','setup','tools/pitch_shift/rubberband-src/build','tools/pitch_shift/rubberband-src','-Dfft=kissfft','-Dresampler=builtin','--wipe'];main()"
meson compile -C tools/pitch_shift/rubberband-src/build
g++ -shared -O2 -I tools/pitch_shift/rubberband-src -o tools/pitch_shift/rubberband_bridge.dll tools/pitch_shift/rubberband_bridge.cpp tools/pitch_shift/rubberband-src/build/librubberband.a -static-libgcc -static-libstdc++
```

The verified source commit built successfully with MinGW GCC 8.1.0 and the
KissFFT/built-in-resampler configuration. The bridge also requires
`libwinpthread-1.dll`, retained beside the bridge DLL. The Rubber Band source
license is in `rubberband-src/COPYING`; bundled KissFFT and Speex license
texts are in `src/ext/*/COPYING`. `libwinpthread-1.dll` is the MinGW
winpthreads runtime and must retain its corresponding MinGW runtime license
when redistributed.

The current local winpthreads DLL is 52,224 bytes with SHA-256
`5bbef249a0d00e2d32c699d0bbe89f714ebeb872b3990a5cbeccb1d89f63e5e8`.
The original DLL build record was not retained; the checksum identifies the
tested local file, rather than proving its upstream build provenance. Native
binaries are excluded from the source repository; build or supply compatible
ones under their original licenses. See `docs/THIRD_PARTY.md`.

Run the strict CPU regression from the project directory with:

```text
runtime\python.exe tools\test_live_pitch_shift.py
```

The regression covers 220 Hz and 800 Hz tones, +2/+12/+14/+20/+24/+36
and -24 semitones, 0.5 s and 2 s windows, finite/length checks, and a
nonstationary burst. It requires frequency errors below 10 cents and burst
start/end differences below 20 ms. It also prints measured 0.5 s p95 timings
for each ratio; these CPU checks alone do not establish realtime device or
perceptual voice quality.

# Vendor files of the public site

The public site (harpor/webapp/site.py) serves every script itself, so its CSP allows scripts from the site only.
These files are copied unchanged from the npm releases below. Each tarball came from registry.npmjs.org and matched
the registry's sha512. site.py checks each file against the SHA-384 in its VENDOR table before it builds, and stops
if one differs. The tunnel page does not use these files.

| File | From | Tarball sha512 (registry dist.integrity) |
|---|---|---|
| `supabase-2.118.0.js` | `@supabase/supabase-js@2.118.0`, `dist/umd/supabase.js` (the same SHA-384 as the SRI hash of this file on jsDelivr) | `sha512-FxI8UXMVC2UG7IkeiI8fuQZqq2oI3JeSuN8MPIC2RLVaoEAxKS0nqE6kii9Ewpq8f9gNPM6git1Y8f0vN/QLFw==` |
| `model-viewer-4.4.0.min.js` | `@google/model-viewer@4.4.0`, `dist/model-viewer.min.js` (it bundles three.js r183) | `sha512-8z1SB5sPlzKoak5mdx/D31brBakPYm3Qi31EfAucHAUrtrHHCbabypkKvSwReYhUpPaxyJ46oT47uWE/hrs1Zw==` |
| `three-0.183.0/draco/draco_wasm_wrapper.js`, `draco_decoder.wasm` | `three@0.183.0`, `examples/jsm/libs/draco/gltf/` (the Draco decoder for glTF) | `sha512-G6SH2jfefIVa2YI4JL2VbgQhrrbp1A8dRc7lr3PW827kdVyaX2RgH6M5FmjmdVFLgSHppyg3OYOZdTfWElle+g==` |
| `three-0.183.0/basis/basis_transcoder.js`, `basis_transcoder.wasm` | `three@0.183.0`, `examples/jsm/libs/basis/` (the KTX2 transcoder) | the same three tarball |
| `licenses/` | each package's LICENSE, and three's notes on the Draco and Basis licenses (Apache 2.0) | the same tarballs |

model-viewer loads the Draco and KTX2 decoders only for compressed GLBs. By default it takes them from www.gstatic.com;
supa.js points it at the two folders above instead. The decoders are those of three.js r183, the three.js that
model-viewer 4.4.0 bundles. The plain-JS Draco decoder (for browsers without WebAssembly) is left out.

To update a file: download the release's tarball from `https://registry.npmjs.org/<package>/-/<name>-<version>.tgz`,
check its sha512 against `dist.integrity` in `https://registry.npmjs.org/<package>/<version>`, copy the file with the
new version in its name, and put its SHA-384 (`openssl dgst -sha384 -binary FILE | base64`) in site.py's VENDOR table.

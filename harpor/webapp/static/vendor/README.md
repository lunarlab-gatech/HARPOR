# Vendor files of the public site

The public site (harpor/webapp/site.py) serves every script itself, so its CSP allows scripts from the site only.
These files are copied unchanged from the npm releases below. Each tarball came from registry.npmjs.org and matched
the registry's sha512. site.py checks each file against the SHA-384 in its VENDOR table before it builds, and stops
if one differs. The tunnel page does not use these files.

| File | From | Tarball sha512 (registry dist.integrity) |
|---|---|---|
| `model-viewer-4.4.0.min.js` | `@google/model-viewer@4.4.0`, `dist/model-viewer.min.js` (it bundles three.js r183) | `sha512-8z1SB5sPlzKoak5mdx/D31brBakPYm3Qi31EfAucHAUrtrHHCbabypkKvSwReYhUpPaxyJ46oT47uWE/hrs1Zw==` |
| `licenses/` | model-viewer and its bundled three.js licenses | the same tarballs |

The gallery accepts uncompressed GLBs and does not ship optional Draco or KTX2 decoders.

To update a file: download the release's tarball from `https://registry.npmjs.org/<package>/-/<name>-<version>.tgz`,
check its sha512 against `dist.integrity` in `https://registry.npmjs.org/<package>/<version>`, copy the file with the
new version in its name, and put its SHA-384 (`openssl dgst -sha384 -binary FILE | base64`) in site.py's VENDOR table.

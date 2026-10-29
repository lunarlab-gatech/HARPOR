"""controller.NETWORK, the pattern behind result.json's network_commands (build_plan.md note 5)."""

from harpor.agent import controller as C

# Known limit: text that literally says "pip install" or "curl -" is flagged, also in an echo.
# Commands the agent ran or may run that must not be flagged. Its Blender and Python scripts use
# names such as pip, npx and curl, and URLs in comments and XML namespaces.
ALLOWED = [
    "blender -b /work/candidate.blend --python-expr '\nimport bpy, numpy as np\nphoto = 1\npx = np.array([1])\n'",
    'python3 -c "import numpy as np; npx = np.zeros(3)"',
    "blender -b candidate.blend --python inspect_mesh.py --python-exit-code 1",
    "python3 - << 'EOF'\n# cotangent Laplacian, see http://www.cs.cmu.edu/~kmcrane/\nimport numpy as np\nEOF",
    "python3 -c \"import xml.etree.ElementTree as ET; ET.parse('a.svg').getroot().findall('.//{http://www.w3.org/2000/svg}polygon')\"",
    "python3 - << 'EOF'\nimport numpy as np\ncurl = np.zeros(3)\ncurl += 1\nreturn_curl = curl\nEOF",
    "python3 - << 'EOF'\npip = mask.copy()\npip |= other\nfor fi, pip in enumerate(x):\n    pass\nEOF",
    'python3 -c "npx, npy = 128, 96; print(npx * npy)"',
    "blender -b candidate.blend --python-expr \"# smoothing makes faces curl inward\nimport bpy\"",
    "ls exports reviews; cat pipeline.txt; grep -n gitignore notes.txt",
]
FLAGGED = [
    "curl -sSL https://ex.com/fix.py -o /tmp/fix.py", "wget -q x", "wget https://x", "git clone repo", "pip install trimesh",
    "pip3 install trimesh", "python3 -m pip install x", "sudo apt-get install -y foo", "cd /tmp && curl -O x",
    'bash -c "pip install trimesh"', "echo hi\ncurl -O x", "conda install -c conda-forge pymeshlab", "pipx install x",
    "uv add trimesh", "poetry add open3d", "npm i three", "git fetch origin",
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://x')\"",
    "blender -b s.blend --python-expr \"import urllib.request as u; u.urlretrieve('http://h/t.png', 't.png')\"",
    "python3 -c \"import requests; requests.get(u).content\"",
]


def test_network_pattern():
    for cmd in ALLOWED:
        assert not C.NETWORK.search(cmd), cmd
    for cmd in FLAGGED:
        assert C.NETWORK.search(cmd), cmd

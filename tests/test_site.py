"""The standalone gallery builds without accounts or a cloud backend."""
from html.parser import HTMLParser

import pytest
from harpor.webapp import site
from test_webapp_demos import hand_demo


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src'))


def test_gallery_build_and_links(tmp_path):
    folder = tmp_path / 'demos'
    hand_demo(folder)
    out = tmp_path / 'site'
    written = site.build(out, demo_dir=folder)
    assert (out / 'index.html').read_bytes() == (out / 'demos.html').read_bytes()
    assert any(name.startswith('demos/') for name in written)
    assert 'static/app.js' not in written and 'static/config.js' not in written
    for page in ('index.html', 'demos.html', 'help.html'):
        text = (out / page).read_text()
        assert 'supabase' not in text.lower() and 'harpor-signin' not in text
        links = Links()
        links.feed(text)
        for link in links.links:
            if link.startswith('#'):
                continue
            assert not link.startswith(('/', 'http:','https:')), link
            assert (out / link.split('#')[0]).is_file(), link
    assert "connect-src 'self' blob:" in (out / '_headers').read_text()
    site.build(out, demo_dir=folder)  # builds can replace earlier gallery output


def test_preserves_gallery_unless_explicitly_removed(tmp_path):
    folder = tmp_path / 'demos'
    hand_demo(folder)
    out = tmp_path / 'site'
    site.build(out, demo_dir=folder)
    before = (out / 'index.html').read_bytes()
    with pytest.raises(site.SiteError, match='earlier build.*has demos'):
        site.build(out)
    assert (out / 'index.html').read_bytes() == before
    site.build(out, no_demos=True)
    assert not (out / 'demos').exists()
    assert 'No demos yet.' in (out / 'index.html').read_text()


def test_refuses_unrelated_directory(tmp_path):
    (tmp_path / 'keep.txt').write_text('keep')
    with pytest.raises(site.SiteError, match='not an earlier build'):
        site.build(tmp_path)
    assert (tmp_path / 'keep.txt').read_text() == 'keep'


def test_replaces_old_platform_build(tmp_path):
    (tmp_path / '.nojekyll').touch()
    (tmp_path / 'static').mkdir()
    (tmp_path / 'static/config.js').write_text('old cloud configuration')
    (tmp_path / 'privacy.html').write_text('old platform privacy')
    site.build(tmp_path)
    assert not (tmp_path / 'static/config.js').exists()
    assert not (tmp_path / 'privacy.html').exists()


def test_vendor_integrity(tmp_path):
    (tmp_path / 'vendor').mkdir()
    for name in site.VENDOR:
        p = tmp_path / 'vendor' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((site.STATIC / 'vendor' / name).read_bytes())
    site.vendor_files(tmp_path)
    (tmp_path / 'vendor/model-viewer-4.4.0.min.js').write_text('changed')
    with pytest.raises(site.SiteError, match='SHA-384'):
        site.vendor_files(tmp_path)


def test_cli_without_configuration(tmp_path):
    assert site.main(['build', '--out', str(tmp_path), '--no-demos']) == 0
    assert (tmp_path / 'index.html').is_file()

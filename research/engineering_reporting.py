"""Offline report helpers over saved figures; no data reads or model fitting."""


def standalone_figure_html(figure):
    """Embed Plotly without parent-page AMD conflicts or external assets.

    A short srcdoc bootstrap decodes the full document inside the frame. The
    encoded payload avoids Quarto extracting a nested head/body into the report
    header; srcdoc also avoids navigation limits on very large data URLs.
    """
    import base64
    import html
    page = figure.to_html(full_html=True, include_plotlyjs=True)
    page = page.replace('<body>', '<body style="margin:0">')
    encoded = base64.b64encode(page.encode('utf-8')).decode('ascii')
    bootstrap = ('<script>const b=Uint8Array.from(atob("'+encoded+'"),c=>c.charCodeAt(0));'
                 'document.write(new TextDecoder().decode(b));document.close();</script>')
    content = html.escape(bootstrap, quote=True)
    height = int(figure.layout.height or 800) + 20
    return (f'<iframe title="Saved portfolio comparison" '
            f'srcdoc="{content}" width="100%" height="{height}" '
            'style="border:0;display:block" loading="eager"></iframe>')

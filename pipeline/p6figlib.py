"""Drawing primitives for the Paper 6 figure deck (native, editable PowerPoint shapes and charts).
Adapted from the Paper 3 and Paper 5 decks: dashed panels with coloured titles, Times New Roman, 18 pt or larger."""
import os, re, copy
from lxml import etree
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Pt

FONT, T = "Times New Roman", 18
INK, INK2, GREY, LIGHT = "16212B", "52514E", "8C8C8C", "E4E3DF"
BLUE, ORANGE, GREEN, RED, PURPLE, DARK, AQUA = "1F6FD1", "E2622B", "1E8C5A", "C0392B", "6250D6", "4D4D4D", "1E8C8C"
LEAK, SOUND, CTRL = "E04B4B", "9A9994", "2E7D32"
MODEL_COL = {"Claude Opus 5.5": "B5532F", "Claude Sonnet 5.5": "E2622B", "GPT-6 Codex": "1E8C5A", "Gemini 4 Hydrogen": "1F6FD1"}


def rgb(h): return RGBColor.from_string(h)


def runs(p, line, size, bold=False, color=INK, italic=False):
    r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.name = FONT; r.font.color.rgb = rgb(color)
    return r


def text(s, x, y, w, h, t, size=T, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = s.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(1)
    for i, line in enumerate(str(t).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; runs(p, line, size, bold, color, italic)
    return tb


def rich(s, x, y, w, h, parts, size=T, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """parts: list of lines; each line a list of (text, bold, color)."""
    tb = s.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(1)
    for i, line in enumerate(parts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        for t_, b, c in line: runs(p, t_, size, b, c)
    return tb


def shape(s, kind, x, y, w, h, fill=None, line=None, lw=1.0, dash=None, rot=0):
    sh = s.shapes.add_shape(kind, Pt(x), Pt(y), Pt(w), Pt(h))
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else: sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
        if dash: sh.line.dash_style = dash
    else: sh.line.fill.background()
    sh.shadow.inherit = False; sh.rotation = rot
    return sh


def box(s, x, y, w, h, t, fill="FFFFFF", line=INK2, size=T, bold=False, color=INK, lw=1.25, kind=None, align=PP_ALIGN.CENTER):
    sh = shape(s, kind or MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, line=line, lw=lw)
    if kind is None: sh.adjustments[0] = 0.16
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Pt(3); tf.margin_top = tf.margin_bottom = Pt(1)
    for i, line_ in enumerate(str(t).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; runs(p, line_, size, bold, color)
    return sh


def container(s, x, y, w, h, title, color, fill="FFFFFF", size=21):
    sh = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, line=color, lw=1.75, dash=MSO_LINE.DASH)
    sh.adjustments[0] = min(0.06, 18 / min(w, h))
    text(s, x + 8, y + 3, w - 16, 28, title, size, True, color)
    return sh


def arrow(s, x1, y1, x2, y2, color=INK2, w=1.75):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x1), Pt(y1), Pt(x2), Pt(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")).set("type", "triangle")
    return c


def line(s, x1, y1, x2, y2, color=INK2, w=1.0, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x1), Pt(y1), Pt(x2), Pt(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    if dash: c.line.dash_style = MSO_LINE.DASH
    return c


def block_arrow(s, x, y, w, h, direction="right", color="A6A5A1"):
    kind = {"right": MSO_SHAPE.RIGHT_ARROW, "down": MSO_SHAPE.DOWN_ARROW, "left": MSO_SHAPE.LEFT_ARROW}[direction]
    return shape(s, kind, x, y, w, h, fill=color)


def padlock(s, x, y, size=20, color=RED):
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x + size * 0.18, y, size * 0.64, size * 0.75, line=color, lw=2.0)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y + size * 0.42, size, size * 0.62, fill=color)


def magnifier(s, x, y, size=26, color=INK2):
    shape(s, MSO_SHAPE.OVAL, x, y, size * 0.7, size * 0.7, fill="FFFFFF", line=color, lw=2.5)
    line(s, x + size * 0.6, y + size * 0.6, x + size, y + size, color=color, w=3.5)


def chip(s, x, y, size=18, color=BLUE, label=None):
    """A small processor icon: a square with pins, used for an LLM."""
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, size, size, fill=color, line=color)
    for i in range(3):
        o = size * (0.22 + 0.28 * i)
        for (a, b, c, d) in ((x + o, y - 4, x + o, y), (x + o, y + size, x + o, y + size + 4), (x - 4, y + o, x, y + o), (x + size, y + o, x + size + 4, y + o)):
            line(s, a, b, c, d, color, 1.5)
    if label: text(s, x, y + size * 0.15, size, size * 0.7, label, 11, True, "FFFFFF", PP_ALIGN.CENTER)


def doc_icon(s, x, y, w=22, h=28, color=INK2):
    shape(s, MSO_SHAPE.FOLDED_CORNER, x, y, w, h, fill="FFFFFF", line=color, lw=1.5)
    for i in range(3): line(s, x + 4, y + 8 + i * 6, x + w - 6, y + 8 + i * 6, color, 1.0)


def wave(s, x, y, w, h, color=BLUE, n=60, seed=3):
    import numpy as np
    """An audio waveform drawn as dense vertical strokes around a centre line."""
    rng = np.random.default_rng(seed); n = 110; xs = np.linspace(x, x + w, n)
    env = 0.55 + 0.45 * np.abs(np.sin(np.linspace(0, 3 * np.pi, n)))
    for i in range(n):
        a = env[i] * (0.4 + 0.6 * rng.random()) * h / 2
        line(s, xs[i], y + h / 2 - a, xs[i], y + h / 2 + a, color, 1.0)


def grid(s, x, y, rows, cols, cell, colours, gap=2):
    for r in range(rows):
        for c in range(cols):
            shape(s, MSO_SHAPE.RECTANGLE, x + c * (cell + gap), y + r * (cell + gap), cell, cell, fill=colours[r][c])


def notes(slide, caption): slide.notes_slide.notes_text_frame.text = caption


# ------------------------------------------------------------------ charts
def num_lit(values):
    el = etree.Element(qn("c:numLit")); etree.SubElement(el, qn("c:formatCode")).text = "General"
    etree.SubElement(el, qn("c:ptCount")).set("val", str(len(values)))
    for i, v in enumerate(values):
        pt = etree.SubElement(el, qn("c:pt")); pt.set("idx", str(i)); etree.SubElement(pt, qn("c:v")).text = repr(float(v))
    return el


def error_bars(series, plus, minus, colour=INK):
    ser = series._element; eb = etree.Element(qn("c:errBars"))
    etree.SubElement(eb, qn("c:errBarType")).set("val", "both"); etree.SubElement(eb, qn("c:errValType")).set("val", "cust")
    etree.SubElement(eb, qn("c:noEndCap")).set("val", "0")
    etree.SubElement(eb, qn("c:plus")).append(num_lit(plus)); etree.SubElement(eb, qn("c:minus")).append(num_lit(minus))
    ln = etree.SubElement(etree.SubElement(eb, qn("c:spPr")), qn("a:ln")); ln.set("w", str(int(Pt(1.75))))
    etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr")).set("val", colour)
    anchor = ser.find(qn("c:cat")); anchor.addprevious(eb)


def style_axis(ax, title=None, lo=None, hi=None, unit=None, fmt=None, gridlines=True, size=T):
    ax.has_major_gridlines = gridlines
    if gridlines: ax.major_gridlines.format.line.color.rgb = rgb("E3E3E3")
    ax.tick_labels.font.size = Pt(size); ax.tick_labels.font.name = FONT; ax.tick_labels.font.color.rgb = rgb(INK)
    ax.format.line.color.rgb = rgb(DARK)
    if lo is not None: ax.minimum_scale = lo
    if hi is not None: ax.maximum_scale = hi
    if unit is not None: ax.major_unit = unit
    if fmt: ax.tick_labels.number_format = fmt; ax.tick_labels.number_format_is_linked = False
    if title:
        ax.has_title = True; tf = ax.axis_title.text_frame; tf.text = title; r = tf.paragraphs[0].runs[0]
        r.font.size = Pt(20); r.font.name = FONT; r.font.bold = False; r.font.color.rgb = rgb(INK)


def style_chart(ch, legend=False, pos=XL_LEGEND_POSITION.BOTTOM):
    ch.font.size = Pt(T); ch.font.name = FONT; ch.font.color.rgb = rgb(INK); ch.has_title = False; ch.has_legend = legend
    if legend: ch.legend.position = pos; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(T); ch.legend.font.name = FONT


def plot_layout(ch, x, y, w, h):
    """Fix the inner plot area (fractions of the chart frame) so neighbouring charts align row by row."""
    pa = ch._chartSpace.find(".//" + qn("c:plotArea")); lay = pa.find(qn("c:layout"))
    if lay is None: lay = etree.SubElement(pa, qn("c:layout")); pa.remove(lay); pa.insert(0, lay)
    for ch_ in list(lay): lay.remove(ch_)
    ml = etree.SubElement(lay, qn("c:manualLayout"))
    for tag, val in (("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"), ("x", x), ("y", y), ("w", w), ("h", h)):
        etree.SubElement(ml, qn("c:" + tag)).set("val", str(val))


def bars(slide, x, y, w, h, cats, vals, colours, horizontal=True, lo=None, hi=None, ci=None, vtitle=None, unit=None, fmt="0.00",
         gap=60, labels=False, show_cats=True, cat_size=T):
    data = CategoryChartData(); data.categories = cats; data.add_series("v", [round(float(v), 4) for v in vals])
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(kind, Pt(x), Pt(y), Pt(w), Pt(h), data).chart; style_chart(ch)
    s = ch.series[0]; s.invert_if_negative = False
    for i, c in enumerate(colours):
        pt = s.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
    ch.plots[0].gap_width = gap
    if ci is not None: error_bars(s, [h_ - v for v, h_ in zip(vals, ci[1])], [v - l_ for v, l_ in zip(vals, ci[0])])
    if labels:
        pl = ch.plots[0]; pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
        dl.font.size = Pt(T); dl.font.name = FONT
    style_axis(ch.value_axis, title=vtitle, lo=lo, hi=hi, unit=unit, fmt=fmt)
    style_axis(ch.category_axis, gridlines=False, size=cat_size)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW if show_cats else XL_TICK_LABEL_POSITION.NONE
    return ch

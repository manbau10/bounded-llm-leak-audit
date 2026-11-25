"""A small Office Math Markup builder.

python-docx has no equation support, so the mathematics is written directly in
the markup Word itself uses. An equation built this way is a real Word
equation. It can be clicked, edited in the equation editor and restyled, which
a picture of an equation cannot.

The expression is written in a compact notation and parsed here.

    x_k        subscript
    x^2        superscript
    x_k^2      both
    @f(a,b)    fraction a over b
    @r(a)      square root of a
    @s(k)(e)   summation over k of e
    |a|        absolute value
    {..}       groups a subexpression

Anything else becomes a run. Letters are set in italic, as mathematics
requires, and digits and operators are set upright.
"""
import re

from docx.oxml.ns import qn
from docx.oxml import OxmlElement

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _el(tag, *children, **attrs):
    e = OxmlElement("m:" + tag)
    for k, v in attrs.items():
        e.set(qn("m:" + k), v)
    for c in children:
        if c is not None:
            e.append(c)
    return e


def _ctrl(italic):
    """The run properties of a mathematical run."""
    rpr = _el("rPr")
    if not italic:
        rpr.append(_el("nor"))
    return rpr


def run(text, italic=None):
    if italic is None:
        italic = bool(re.fullmatch(r"[A-Za-z]+", text.strip())) and \
            text.strip() not in FUNCTIONS
    r = _el("r", _ctrl(italic))
    t = OxmlElement("m:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


FUNCTIONS = {"log", "exp", "tanh", "softmax", "ReLU", "BN", "max", "min",
             "acc", "conf", "sign", "Attention", "STFT", "round", "eff", "H", "mean"}

GREEK = {"eps": "ε", "sigma": "σ", "Sigma": "Σ",
         "tau": "τ", "beta": "β", "alpha": "α",
         "Delta": "Δ", "odot": "⊙", "times": "×",
         "in": "∈", "R": "ℝ", "approx": "≈",
         "leq": "≤", "cdot": "·", "minus": "−",
         "rho": "ρ", "pi": "π", "lambda": "λ", "mu": "μ", "geq": "≥", "hat": "̂", "larr": "←"}


def _tokenize(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            out.append(("sym", GREEK.get(m.group(1), m.group(1))))
            i += m.end()
        elif c == "@":
            m = re.match(r"@([frs])", s[i:])
            out.append(("op", m.group(1)))
            i += m.end()
        elif c in "{}()|,":
            out.append(("punct", c))
            i += 1
        elif c in "_^":
            out.append(("script", c))
            i += 1
        elif c == " ":
            out.append(("space", " "))
            i += 1
        else:
            m = re.match(r"[A-Za-z]+|[0-9.]+|[^A-Za-z0-9.{}()|,_^ \\@]+", s[i:])
            out.append(("text", m.group(0)))
            i += m.end()
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse_seq(self, stop=()):
        items = []
        while self.i < len(self.t):
            kind, val = self.peek()
            if kind == "punct" and val in stop:
                break
            items.append(self.parse_one())
        return items

    def parse_group(self):
        """A braced group, a parenthesised group, or a single atom."""
        kind, val = self.peek()
        if kind == "punct" and val == "{":
            self.next()
            items = self.parse_seq(stop=("}",))
            if self.peek()[1] == "}":
                self.next()
            return items
        return [self.parse_one(raw=True)]

    def parse_one(self, raw=False):
        kind, val = self.next()
        if kind == "op":
            if val == "f":
                num, den = self.parse_group(), self.parse_group()
                return _el("f", _el("fPr", _el("type", val="bar")),
                           _el("num", *num), _el("den", *den))
            if val == "r":
                body = self.parse_group()
                return _el("rad", _el("radPr", _el("degHide", val="1")),
                           _el("deg"), _el("e", *body))
            if val == "s":
                sub, body = self.parse_group(), self.parse_group()
                nary = _el("nary",
                           _el("naryPr", _el("chr", val="∑"),
                               _el("limLoc", val="undOvr"),
                               _el("supHide", val="1")),
                           _el("sub", *sub), _el("sup"), _el("e", *body))
                return nary
        if kind == "punct" and val == "|":
            body = self.parse_seq(stop=("|",))
            if self.peek()[1] == "|":
                self.next()
            return _el("d", _el("dPr", _el("begChr", val="|"),
                                _el("endChr", val="|")), _el("e", *body))
        if kind == "punct" and val == "(":
            body = self.parse_seq(stop=(")",))
            if self.peek()[1] == ")":
                self.next()
            return _el("d", _el("dPr", _el("begChr", val="("),
                                _el("endChr", val=")")), _el("e", *body))
        if kind == "punct" and val == "{":
            body = self.parse_seq(stop=("}",))
            if self.peek()[1] == "}":
                self.next()
            base = _el("e", *body)
            return self._scripts(base, wrap=True)
        if kind == "space":
            return run(" ", italic=False)
        if kind == "sym":
            base = run(val, italic=False)
        elif kind == "punct":
            base = run(val, italic=False)
        else:
            base = run(val)
        if raw:
            return base
        return self._scripts(base)

    def _scripts(self, base, wrap=False):
        kind, val = self.peek()
        if kind != "script":
            return base if not wrap else _el("e", base) if False else base
        sub = sup = None
        while self.peek()[0] == "script":
            _, which = self.next()
            grp = self.parse_group()
            if which == "_":
                sub = grp
            else:
                sup = grp
        if sub is not None and sup is not None:
            return _el("sSubSup", _el("e", base), _el("sub", *sub),
                       _el("sup", *sup))
        if sub is not None:
            return _el("sSub", _el("e", base), _el("sub", *sub))
        return _el("sSup", _el("e", base), _el("sup", *sup))


def omath(expr):
    """Build one m:oMath element from the compact notation."""
    p = _Parser(_tokenize(expr))
    return _el("oMath", *p.parse_seq())


def add_equation(paragraph, expr, number=None):
    """Place an equation in a paragraph, with an optional number on the right.

    The equation is centred and the number is set flush right, which is the
    layout every journal in this field asks for.
    """
    para = paragraph._p
    if number is not None:
        # a centre tab at the middle of the text column and a right tab at its
        # edge, so one line holds both the equation and its number
        from docx.enum.text import WD_TAB_ALIGNMENT
        tabs = paragraph.paragraph_format.tab_stops
        from docx.shared import Cm
        tabs.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER)
        tabs.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
        paragraph.add_run("\t")
    para.append(omath(expr))
    if number is not None:
        paragraph.add_run("\t(%s)" % number)
    return paragraph

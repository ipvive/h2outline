import math
import cmath
from cmath import sinh, acosh, cosh, sin, log, exp, pi
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import copy
import re

Span = namedtuple("Span", ["doc", "start", "stop"])
Outline = namedtuple("Outline", ["loc", "itemtext", "children"])


def outline(loc, itemtext=None, children=None):
    return Outline(loc, itemtext, children)


def md2dot(span):
    t = parse_outline(span)
    return emit_dot(t)


def parse_outline(span):
    def parse1(span):
        if is_single_line(span):
            return Outline(location(span), itemtext(span), None)
        else:
            return Outline(location(span), itemtext(span), parse_outline(Span(span.doc, span.start + 1, span.stop)))

    indent = re.match(r"^\s*", span.doc[span.start]).group(0)
    sibling_starts = [n for n in range(span.start, span.stop)
                      if has_indent(span.doc[n], indent)]
    siblings = [parse1(Span(span.doc, n, nextn))
                for n, nextn in zip(sibling_starts, sibling_starts[1:] + [span.stop])]
    return siblings


def is_single_line(span):
    return span.start + 1 >= span.stop


def has_indent(line, indent):
    return line.startswith(indent + "*")


def itemtext(span):
    line = span.doc[span.start]
    i = line.index("* ")
    return line[i + 2:]


def location(span):
    if is_single_line(span):
        return f"L{span.start + 1}"
    else:
        return f"L{span.start + 1}-L{span.stop}"


def l(n):
    """triangle with angles 0, pi/2, pi/(2*n) has finite edge with length l."""
    if n < 3:
        n = 3
    return acosh(1/sin(pi/n))


def t(l):
    """translate origin north by a distance if l."""
    return [[cosh(l/2), sinh(l/2)], [sinh(l/2), cosh(l/2)]]


def r(i, n):
    """rotate i/n revolutions about origin."""
    theta = pi * i / n
    return [[exp(1j * theta), 0], [0, exp(-1j * theta)]]


def compose(l):
    def compose1(m, n):
        return [[m[i][0] * n[0][j] + m[i][1] * n[1][j] for j in (0, 1)] for i in (0, 1)]
    if len(l) == 0:
        return [[1, 0], [0, 1]]
    else:
        return compose1(l[0], compose(l[1:]))


def draw_tree(tree, pmat):
    """draw the points and edges of tree, translated by pmat."""
    draw_node(pmat, tree)
    n = 1 + len(tree.children)
    for i, c in enumerate(tree.children):
        ri = r(1 + i, n)
        draw_line_and_subtree(tree, c, compose([pmat, ri]))


def draw_line_and_subtree(parent, child, pmat):
    n = 1 + len(parent.children)
    p = [child]
    m = 1 + len(child.children)
    while m == 2:
        child = child.children[0]
        p += [child]
        m = 1 + len(child.children)
    ls = np.linspace(0, l(n) + l(m), 1 + len(p))
    for ll, c in zip(ls[1:], p[:-1]):
        draw_node(compose([pmat, t(ll)]), c)
    draw_zs(seg(pmat, ls[-1]))
    draw_tree(child, compose([pmat, t(ls[-1]), r(1, 2)]))


def z(p):
    o = [0, 1]
    return np.dot(p, o)


def seg(pm, l, n=50):
    ls = np.linspace(0, l, n)
    pms = [compose([pm, t(ll)]) for ll in ls]
    zs = [z(m) for m in pms]
    return zs


def draw_zs(zs):
    ns = [z[0] / z[1] for z in zs]
    xs = np.array([z.imag for z in ns])
    ys = np.array([z.real for z in ns])
    plt.plot(xs, ys)


def draw_node(pm, node=None):
    zz = z(pm)
    n = zz[0] / zz[1]
    l, = plt.plot(n.imag, n.real, 'o', picker=True)
    l.node = node
    if node:
        plt.text(n.imag, n.real, node.label)


def draw_boundary():
    theta = np.linspace(0, 2*pi)
    c = plt.plot(np.sin(theta), np.cos(theta))
    plt.gca().set_aspect(1.0)
    return c


class OTree:
    def __init__(self, o, p=None):
        self.o = o
        self.p = p
        self.c = [OTree(c, self) for c in self.o.children or []]

    @property
    def children(self):
        return self.c

    @property
    def parent(self):
        return self.p

    @property
    def label(self):
        return self.o.loc + self.o.itemtext.split(" ")[0]

    @property
    def node(self):
        return self


def make_OTree(os):
    o = Outline("", ".", os)
    return OTree(o)


def tocenter(z):
    r, th = cmath.polar(z)
    th = -th
    rot0 = [[exp(-1j * th/2), 0], [0, exp(1j * th/2)]]
    trans = t(2 * math.atanh(-r))
    rot1 = [[exp(1j * th/2), 0], [0, exp(-1j * th/2)]]
    res = compose([rot0, trans, rot1])
    return res

def normalize(m):
    ar = 0.5 * (m[0][0] + m[1][1]).real
    ai = 0.5 * (m[0][0] - m[1][1]).imag
    br = 0.5 * (m[0][1] + m[1][0]).real
    bi = 0.5 * (m[0][1] - m[1][0]).imag
    m = [[ar + ai*1j, br + bi*1j], [br - bi*1j, ar - ai*1j]]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    f = 1 / cmath.sqrt(abs(det))
    m = [[f*m[0][0], f*m[0][1]], [f*m[1][0], f*m[1][1]]]
    return m

def transport(z0, z1):
    m0 = tocenter(z0)
    p1 = [z1,1]
    m0p1 = np.dot(m0,p1)
    m1 = tocenter(m0p1[0] / m0p1[1])
    m2 = tocenter(-z0)
    return compose([m2, m1, m0])


class Fig:
    def __init__(self, tree):
        self.tree = tree
        #self.m = compose([])
        self.m = compose([])
        self.dragpos = None
        self.dragm = None
        plt.gcf().canvas.mpl_connect('button_press_event', self.onpress)
        plt.gcf().canvas.mpl_connect('motion_notify_event', self.onmove)
        plt.gcf().canvas.mpl_connect('button_release_event', self.onrelease)
        plt.gcf().canvas.mpl_connect('pick_event', self.onpick)

    def draw(self):
        plt.clf()
        draw_boundary()
        draw_tree(self.tree, self.m)
        plt.gcf().canvas.draw()
        plt.show()

    def onmove(self, event):
        if self.dragpos:
            # print('onmove', event)
            self.ondrag((event.xdata, event.ydata), self.dragpos, self.dragm)
        self.draw()

    def onpress(self, event):
        if event.button == 1:
            # print('onpress', event)
            self.dragpos = (event.xdata, event.ydata)
            self.dragm = copy.deepcopy(self.m)

    def onrelease(self, event):
        if self.dragpos:
            # print('onrelease', event)
            dragpos = self.dragpos
            dragm = self.dragm
            self.dragpos = None
            self.dragm = None
            self.ondrag((event.xdata, event.ydata), dragpos, dragm)

    def ondrag(self, curpos, dragpos, dragm):
        # print('ondrag', dragpos, curpos)
        x0, y0 = dragpos
        x1, y1 = curpos
        z0, z1 = y0 + 1j*x0, y1 + 1j*x1 
        if abs(z0) >= 0.99 or abs(z1) >= 0.99:
            self.dragpos = None
            self.dragm = None
            return
        tm = transport(z1, z0)
        # print(f"tm={tm}")
        self.m = normalize(compose([tm, dragm]))
        self.draw()
        plt.gcf().canvas.flush_events()

    def onpick(self, event):
        node = event.artist.node
        print('onpick', node.o.loc, node.o.itemtext)
        self.draw()



with open("README.md") as f:
    lines = f.readlines()
span = Span(lines, 0, len(lines))
o = parse_outline(span)

tree = make_OTree(o)
f = Fig(tree)
f.draw()

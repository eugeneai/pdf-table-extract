import sys
import os

DEBUG = False

if DEBUG:
    import random
from numpy import array, fromstring, ones, zeros, uint8, diff, where, sum, delete, frombuffer, reshape, all, any
import numpy

if DEBUG:
    import matplotlib
    matplotlib.use('AGG')
    from matplotlib.image import imsave

from xml.dom.minidom import getDOMImplementation
import json
import csv
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Poppler', '0.18')
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk, Poppler  #, Glib
import cairo


def interact(locals):
    import code
    code.InteractiveConsole(locals=locals).interact()


class PopplerProcessor(object):
    """Class for processing PDF. That's simple.
    It does two functions.
    1. Renders a page as a PNM graphics, and
    2. Get text in a rectangular bounding box.
    """

    def __init__(self, filename, **kwargs):
        """Opens a document denoted by filename.
        """
        self.filename = os.path.abspath(filename)
        self.document = Poppler.Document.new_from_file("file:" + self.filename,
                                                       None)
        self.page_num = self.document.get_n_pages()
        self.resolution = 300
        self.greyscale_threshold = int(kwargs.get("greyscale_thresholds",
                                                  25)) * 255.0 / 100.0
        self.layout = None

    def get_page(self, index):
        if index < 0 or index >= self.page_num:
            raise IndexError("page number is out of bounds")
        page = self.document.get_page(index)
        if self.layout != None:
            #Glib.free(self.layout)
            # Do we need freeing elements of the list # FIXME
            self.layout = None
        self.text = page.get_text()
        self.attributes=page.get_text_attributes()
        l = page.get_text_layout()
        if l[0]:
            self.layout = l[1]
        return page

    def get_image(self, index):
        page = self.get_page(index)
        dpi = self.resolution
        scale = 1
        width, height = [int(x) for x in page.get_size()]
        d = self.scale = dpi / 72.
        self.frac_scale = 1 / d
        pxw, pxh = int(width * d), int(height * d)
        surface = cairo.ImageSurface(
            # data,
            cairo.FORMAT_ARGB32,
            pxw,
            pxh)

        context = cairo.Context(surface)
        context.scale(d, d)

        context.save()
        page.render(context)
        context.restore()

        pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, pxw, pxh)
        # surface.write_to_png("page.png")
        data = frombuffer(pixbuf.get_pixels(), dtype=uint8)
        R = data[0::4]
        G = data[1::4]
        B = data[2::4]
        A = data[3::4]
        C = (R * 34. + G * 56. + B * 10.) / 100. # Convert to gray

        C = C.astype(uint8)

        nd = zeros(C.shape, dtype=uint8)
        nd[:] = C
        nd[A <= self.greyscale_threshold] = 255
        nd = nd.reshape((pxh, pxw))
        # imsave('nomask.png', nd)
        return nd, page

    def print_rect(self, msg=None, r=None, page=None):
        """Used for debugging.
        """
        if None in [r, page]:
            raise ValueError("r and page arguments are required")
        x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        print(msg, x, y, w, h, "---", x1, y1, x2, y2)
        width, height = [int(x) for x in page.get_size()]
        print(msg, x, height - y, w, h, "---", x1, height - y1, x2,
              height - y2)

    def within(self, a, b, pad=0):
        """Is Rectangle b within Rectangle a, i.e. b is in a.

        Arguments:
        - `a`, `b` : The rectangles;
        - `pad` : Additional space.
        """
        if b.x1+pad < a.x1: return False
        if b.y1+pad < a.y1: return False
        if b.x2-pad > a.x2: return False
        if b.y2-pad > a.y2: return False
        return True

    def rexpand(self, rect, layout, pad=0):
        """Make rectangle rect include layout

        Arguments:
        - `rect`: Adjustable Rectangle;
        - `layout`: Rectangle to be included in rect.
        """

        r, l = rect, layout
        if r.x1 > l.x1: r.x1 = l.x1-pad
        if r.y1 > l.y1: r.y1 = l.y1-pad
        if r.x2 < l.x2: r.x2 = l.x2+pad
        if r.y2 < l.y2: r.y2 = l.y2+pad

    def get_text(self, page, x, y, w, h):
        width, height = [int(x) for x in page.get_size()]
        fc = self.frac_scale
        x, y, w, h = (z * fc for z in [x, y, w, h])
        rect = Poppler.Rectangle()
        rect.x1, rect.y1 = x, y
        rect.x2, rect.y2 = x + w, y + h
        assert rect.x1<=rect.x2
        assert rect.y1<=rect.y2

        # Could not make it work correctly # FIXME
        # txt = page.get_text_for_area(rect)
        # attrs = page.get_text_attributes_for_area(rect)

        r = Poppler.Rectangle()
        r.x1 = r.y1 = 1e10
        r.x2 = r.y2 = -1e10
        chars=[]
        for k,l in enumerate(self.layout):
            if self.within(rect, l, pad=1):
                self.rexpand(r, l, pad=0.5)
                chars.append(self.text[k])
        txt="".join(chars)

        # txt = page.get_text_for_area(r) # FIXME

        return txt, r

    def get_rectangles_for_page(self, page):
        """Return all rectangles for all letters in the page..
        Used for debugging.

        Arguments:
        - `page`: referece to page
        """
        layout=self.layout
        if layout == None:
            raise RuntimeError("page is not chosen")

        answer = [(r.x1,r.y1,r.x2,r.y2) for r in layout]
        return answer


def colinterp(a, x):
    """Interpolates colors"""
    l = len(a) - 1
    i = min(l, max(0, int(x * l)))
    (u, v) = a[i:i + 2, :]
    return u - (u - v) * ((x * l) % 1.0)


colarr = array(
    [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]])


def col(x, colmult=1.0):
    """colors"""
    return colinterp(colarr, (colmult * x) % 1.0) / 2

def process_page(infile,
                 pgs,
                 outfilename=None,
                 greyscale_threshold=25,
                 page=None,
                 crop=None,
                 line_length=0.5,
                 bitmap_resolution=300,
                 name=None,
                 pad=2,
                 white=None,
                 black=None,
                 bitmap=False,
                 checkcrop=False,
                 checklines=False,
                 checkdivs=False,
                 checkcells=False,
                 checkall=False,
                 checkletters=False,
                 whitespace="normalize",
                 boxes=False,
                 encoding="utf8"):

    if checkall:
        checkcrop = True
        checklines = True
        checkdivs = True
        checkcells = True
        checkletters = True

    outfile = outfilename if outfilename else "output"
    pdfdoc = PopplerProcessor(infile)
    page = page or []
    (pg, frow, lrow) = (list(map(int, (pgs.split(":")))) + [None, None])[0:3]
    pdfdoc.resolution = bitmap_resolution
    pdfdoc.greyscale_threshold = greyscale_threshold

    data, page = pdfdoc.get_image(pg - 1)  # Page numbers are 0-based.

    #-----------------------------------------------------------------------
    # image load section.

    height, width = data.shape[:2]  # If not to reduce to gray, the shape will be (,,3) or (,,4).

    pad = int(pad)
    height += pad * 2
    width += pad * 2

    # reimbed image with a white pad.
    bmp = ones((height, width), dtype=bool)

    thr = int(255.0 * greyscale_threshold / 100.0)

    bmp[pad:height - pad, pad:width - pad] = (data[:, :] > thr)


    # Set up Debuging image.
    img = zeros((height, width, 3), dtype=uint8)

    # img[:, :, :] = bmp * 255 # In case of colored input image

    img[:, :, 0] = bmp * 255
    img[:, :, 1] = bmp * 255
    img[:, :, 2] = bmp * 255

    if checkdivs or checkcells or checkletters:
        imgfloat = img.astype(float)

    if checkletters:  # Show bounding boxes for each text object.
        img = (imgfloat/2.).astype(uint8)
        rectangles=pdfdoc.get_rectangles_for_page(pg)
        lrn=len(rectangles)
        for k,r in enumerate(rectangles):
            x1,y1,x2,y2 = [int(bitmap_resolution* float(k)/72.)+pad for k in r]
            img[y1:y2, x1:x2] += col(random.random()).astype(uint8)
        imsave("letters.png", img)


    #-----------------------------------------------------------------------
    # Find bounding box.
    t = 0

    while t < height and all(bmp[t, :]):
        t = t + 1
    if t > 0:
        t = t - 1

    b = height - 1
    while b > t and all(bmp[b, :]):
        b = b - 1
    if b < height - 1:
        b = b + 1

    l = 0
    while l < width and all(bmp[:, l]):
        l = l + 1
    if l > 0:
        l = l - 1

    r = width - 1
    while r > l and all(bmp[:, r]):
        r = r - 1
    if r < width - 1:
        r = r + 1

# Mark bounding box.
    bmp[t, :] = False
    bmp[b, :] = False
    bmp[:, l] = False
    bmp[:, r] = False

    def boxOfString(x, p):
        s = x.split(":")
        if len(s) < 4:
            raise ValueError("boxes have format left:top:right:bottom[:page]")
        return ([bitmap_resolution * float(x) + pad for x in s[0:4]] +
                [p if len(s) < 5 else int(s[4])])

# translate crop to paint white.

    whites = []
    if crop:
        (l, t, r, b, p) = boxOfString(crop, pg)
        whites.extend([(0, 0, l, height, p), (0, 0, width, t, p),
                       (r, 0, width, height, p), (0, b, width, height, p)])

# paint white ...
    if white:
        whites.extend([boxOfString(b, pg) for b in white])

    for (l, t, r, b, p) in whites:
        if p == pg:
            bmp[t:b + 1, l:r + 1] = 1
            img[t:b + 1, l:r + 1] = [255, 255, 255]

# paint black ...
    if black:
        for b in black:
            (l, t, r,
             b) = [bitmap_resolution * float(x) + pad for x in b.split(":")]
            bmp[t:b + 1, l:r + 1] = 0
            img[t:b + 1, l:r + 1] = [0, 0, 0]

    if checkcrop:
        imsave("crop-" + outfile + ".png", img)

#-----------------------------------------------------------------------
# Line finding section.
#
# Find all vertical or horizontal lines that are more than lthresh
# long, these are considered lines on the table grid.

    lthresh = int(line_length * bitmap_resolution)
    vs = zeros(width, dtype=uint8)

    for i in range(width):
        dd = diff(where(bmp[:, i])[0])
        if len(dd) > 0:
            v = max(dd)
            if v > lthresh:
                vs[i] = 1
        else:
            # it was a solid black line.
            if all(bmp[0, i]) == 0:
                vs[i] = 1
    vd = (where(diff(vs[:]))[0] + 1)

    hs = zeros(height, dtype=uint8)
    for j in range(height):
        dd = diff(where(bmp[j, :])[0])
        if len(dd) > 0:
            h = max(dd)
            if h > lthresh:
                hs[j] = 1
        else:
            # it was a solid black line.
            if all(bmp[j, 0]) == 0:
                hs[j] = 1
    hd = (where(diff(hs[:]))[0] + 1)

    #-----------------------------------------------------------------------
    # Look for dividors that are too large.
    maxdiv = 10
    i = 0

    while i < len(vd):
        if vd[i + 1] - vd[i] > maxdiv:
            vd = delete(vd, i)
            vd = delete(vd, i)
        else:
            i = i + 2

    j = 0
    while j < len(hd):
        if hd[j + 1] - hd[j] > maxdiv:
            hd = delete(hd, j)
            hd = delete(hd, j)
        else:
            j = j + 2

    if checklines:
        for i in vd:
            img[:, i] = [255, 0, 0]  # red

        for j in hd:
            img[j, :] = [0, 0, 255]  # blue
        imsave("lines-" + outfile + ".png", img)

        #-----------------------------------------------------------------------
        # divider checking.
        #
        # at this point vd holds the x coordinate of vertical  and
        # hd holds the y coordinate of horizontal divider tansitions for each
        # vertical and horizontal lines in the table grid.

    def isDiv(a, l, r, t, b):
        # if any col or row (in axis) is all zeros ...
        return sum(sum(bmp[t:b, l:r], axis=a) == 0) > 0

    if checkdivs:
        img = (imgfloat / 2).astype(uint8)
        for j in range(0, len(hd), 2):
            for i in range(0, len(vd), 2):
                if i > 0:
                    (l, r, t, b) = (vd[i - 1], vd[i], hd[j], hd[j + 1])
                    img[t:b, l:r, 1] = 192
                    if isDiv(1, l, r, t, b):
                        img[t:b, l:r, 0] = 0
                        img[t:b, l:r, 2] = 255

                if j > 0:
                    (l, r, t, b) = (vd[i], vd[i + 1], hd[j - 1], hd[j])
                    img[t:b, l:r, 1] = 128
                    if isDiv(0, l, r, t, b):
                        img[t:b, l:r, 0] = 255
                        img[t:b, l:r, 2] = 0
        imsave("divs-" + outfile + ".png", img)

        #-----------------------------------------------------------------------
        # Cell finding section.
        # This algorithum is width hungry, and always generates rectangular
        # boxes.

    cells = []
    touched = zeros((len(hd), len(vd)), dtype=bool)
    j = 0
    while j * 2 + 2 < len(hd):
        i = 0
        while i * 2 + 2 < len(vd):
            u = 1
            v = 1
            if not touched[j, i]:
                while 2+(i+u)*2 < len(vd) and \
                    not isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1],
                       hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ] ):
                    u = u + 1
                bot = False
                while 2 + (j + v) * 2 < len(hd) and not bot:
                    bot = False
                    for k in range(1, u + 1):
                        bot |= isDiv(1, vd[2 * (i + k) - 1], vd[2 * (i + k)],
                                     hd[2 * (j + v)], hd[2 * (j + v) + 1])
                    if not bot:
                        v = v + 1
                cells.append((i, j, u, v))
                touched[j:j + v, i:i + u] = True
            i = i + 1
        j = j + 1

    if checkcells:
        nc = len(cells) + 0.
        img = (imgfloat / 2.).astype(uint8)
        for k in range(len(cells)):
            (i, j, u, v) = cells[k]
            (l, r, t, b) = (vd[2 * i + 1], vd[2 * (i + u)], hd[2 * j + 1],
                            hd[2 * (j + v)])
            img[t:b, l:r] += col(k / nc).astype(uint8)

        imsave("cells-" + outfile + ".png", img)

        #-----------------------------------------------------------------------
        # fork out to extract text for each cell.

    def getCell(_coordinate, img=None):
        (i, j, u, v) = _coordinate
        (l, r, t, b) = (vd[2 * i + 1], vd[2 * (i + u)], hd[2 * j + 1],
                        hd[2 * (j + v)])
        ret, rect = pdfdoc.get_text(page, l - pad, t - pad, r - l, b - t)

        if type(img)!=type(None) and checkletters:
            (x1,y1,x2,y2) = [int(bitmap_resolution * float(rrr)/72+pad) for rrr in [rect.x1,rect.y1,rect.x2,rect.y2]]
            img[y1:y2,x1:x2] += col(random.random()).astype(uint8)

        return (i, j, u, v, pg, ret)

    if checkletters:
        img = (imgfloat / 2.).astype(uint8)

    if boxes:
        cells = [x + (pg,
                      "", ) for x in cells
                 if (frow == None or (x[1] >= frow and x[1] <= lrow))]
    else:
        cells = [getCell(x, img) for x in cells
                 if (frow == None or (x[1] >= frow and x[1] <= lrow))]
    if checkletters:
        imsave("text-locations.png", img)

    return cells

#-----------------------------------------------------------------------
#output section.


def output(cells,
           pgs,
           cells_csv_filename=None,
           cells_json_filename=None,
           cells_xml_filename=None,
           table_csv_filename=None,
           table_html_filename=None,
           table_list_filename=None,
           infile=None,
           name=None,
           output_type=None):

    output_types = [
        dict(filename=cells_csv_filename,
             function=o_cells_csv), dict(filename=cells_json_filename,
                                         function=o_cells_json),
        dict(filename=cells_xml_filename,
             function=o_cells_xml), dict(filename=table_csv_filename,
                                         function=o_table_csv),
        dict(filename=table_html_filename,
             function=o_table_html), dict(filename=table_list_filename,
                                          function=o_table_list)
    ]

    for entry in output_types:
        if entry["filename"]:
            if entry["filename"] != sys.stdout:
                outfile = open(entry["filename"], 'w')
            else:
                outfile = sys.stdout

            entry["function"](cells,
                              pgs,
                              outfile=outfile,
                              name=name,
                              infile=infile,
                              output_type=output_type)

            if entry["filename"] != sys.stdout:
                outfile.close()


def o_cells_csv(cells,
                pgs,
                outfile=None,
                name=None,
                infile=None,
                output_type=None):
    outfile = outfile or sys.stdout
    csv.writer(outfile, dialect='excel').writerows(cells)


def o_cells_json(cells,
                 pgs,
                 outfile=None,
                 infile=None,
                 name=None,
                 output_type=None):
    """Output JSON formatted cell data"""
    outfile = outfile or sys.stdout
    #defaults
    infile = infile or ""
    name = name or ""

    json.dump({
        "src": infile,
        "name": name,
        "colnames": ("x", "y", "width", "height", "page", "contents"),
        "cells": cells
    }, outfile)


def o_cells_xml(cells,
                pgs,
                outfile=None,
                infile=None,
                name=None,
                output_type=None):
    """Output XML formatted cell data"""
    outfile = outfile or sys.stdout
    #defaults
    infile = infile or ""
    name = name or ""

    def _lambda(a):
        return x.setAttribute(*a)

    doc = getDOMImplementation().createDocument(None, "table", None)
    root = doc.documentElement
    if infile:
        root.setAttribute("src", infile)
    if name:
        root.setAttribute("name", name)
    for cl in cells:
        x = doc.createElement("cell")
        map(_lambda, zip("xywhp", map(str, cl)))
        if cl[5] != "":
            x.appendChild(doc.createTextNode(cl[5]))
        root.appendChild(x)
    outfile.write(doc.toprettyxml())


def table_to_list(cells, pgs):
    """Output list of lists"""
    l = [0, 0, 0]
    for (i, j, u, v, pg, value) in cells:
        r = [i, j, pg]
        l = [max(x) for x in zip(l, r)]

    tab = [[["" for x in range(l[0] + 1)] for x in range(l[1] + 1)]
           for x in range(l[2] + 1)]
    for (i, j, u, v, pg, value) in cells:
        tab[pg][j][i] = value

    return tab


def o_table_csv(cells,
                pgs,
                outfile=None,
                name=None,
                infile=None,
                output_type=None):
    """Output CSV formatted table"""
    outfile = outfile or sys.stdout
    tab = table_to_list(cells, pgs)
    for t in tab:
        csv.writer(outfile, dialect='excel').writerows(t)


def o_table_list(cells,
                 pgs,
                 outfile=None,
                 name=None,
                 infile=None,
                 output_type=None):
    """Output list of lists"""
    outfile = outfile or sys.stdout
    tab = table_to_list(cells, pgs)
    print(tab)


def o_table_html(cells,
                 pgs,
                 outfile=None,
                 output_type=None,
                 name=None,
                 infile=None):
    """Output HTML formatted table"""

    oj = 0
    opg = 0
    doc = getDOMImplementation().createDocument(None, "table", None)
    root = doc.documentElement
    if (output_type == "table_chtml"):
        root.setAttribute("border", "1")
        root.setAttribute("cellspaceing", "0")
        root.setAttribute("style", "border-spacing:0")
    nc = len(cells)
    tr = None
    for k in range(nc):
        (i, j, u, v, pg, value) = cells[k]
        if j > oj or pg > opg:
            if pg > opg:
                s = "Name: " + name + ", " if name else ""
                root.appendChild(doc.createComment(s + ("Source: %s page %d." %
                                                        (infile, pg))))
            if tr:
                root.appendChild(tr)
            tr = doc.createElement("tr")
            oj = j
            opg = pg
        td = doc.createElement("td")
        if value != "":
            td.appendChild(doc.createTextNode(value))
        if u > 1:
            td.setAttribute("colspan", str(u))
        if v > 1:
            td.setAttribute("rowspan", str(v))
        if output_type == "table_chtml":
            td.setAttribute("style", "background-color: #%02x%02x%02x" %
                            tuple(128 + col(k / (nc + 0.))))
        tr.appendChild(td)
    root.appendChild(tr)
    outfile.write(doc.toprettyxml())

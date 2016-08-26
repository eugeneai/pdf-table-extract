import sys
import os

from numpy import array, ones, zeros, uint8, diff, where, sum
from numpy import delete, frombuffer, reshape, all, any
import numpy
import copy

#from xml.dom.minidom import getDOMImplementation
from lxml import etree
import json
import csv
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Poppler', '0.18')
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk, Poppler
import cairo
import collections


def interact(locals):
    """Starts interactive console,
    Used for debugging.
    """
    import code
    code.InteractiveConsole(locals=locals).interact()


class PopplerProcessor(object):
    """Class for processing PDF.
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
        self.table_chars = set()  # Contains indexes of chars belonging to tables.
        # Used to recognize text of paragraphs.

    def get_page(self, index):
        if index < 0 or index >= self.page_num:
            raise IndexError("page number is out of bounds")
        page = self.document.get_page(index)
        self.text = page.get_text()
        self.attributes = page.get_text_attributes()
        self.table_chars = set()
        l = page.get_text_layout()
        if l[0]:
            self.layout = l[1]
        else:
            self.layout = None

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
        C = (R * 34. + G * 56. + B * 10.) / 100.  # Convert to gray

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

    def inside(self, a, b, pad=0):
        """Check if Rectangle a is mostly inside Rectangle b.

        Arguments:
        - `a`, `b` : The rectangles;
        - `pad` : Additional space. (IGNORED)
        """
        ay = (a.y1 + a.y2) / 2.
        return a.x1 < b.x2 and a.x2 > b.x1 and ay < b.y2 and ay > b.y1

    def overlap(self, a, b, pad=0):
        """Check if Rectangle a and Rectangle b overlaps.

        Arguments:
        - `a`, `b` : The rectangles;
        - `pad` : Additional space. (IGNORED)
        """
        return a.x1 < b.x2 and a.x2 > b.x1 and a.y1 < b.y2 and a.y2 > b.y1

    def rexpand(self, rect, layout, pad=0):
        """Make rectangle rect include layout

        Arguments:
        - `rect`: Adjustable Rectangle;
        - `layout`: Rectangle to be included in rect.
        """

        r, l = rect, layout
        if r.x1 > l.x1: r.x1 = l.x1 - pad
        if r.y1 > l.y1: r.y1 = l.y1 - pad
        if r.x2 < l.x2: r.x2 = l.x2 + pad
        if r.y2 < l.y2: r.y2 = l.y2 + pad

    def get_text(self, page, x, y, w, h):
        width, height = [int(x) for x in page.get_size()]
        fc = self.frac_scale
        x, y, w, h = (z * fc for z in [x, y, w, h])
        rect = Poppler.Rectangle()
        rect.x1, rect.y1 = x, y
        rect.x2, rect.y2 = x + w, y + h
        assert rect.x1 <= rect.x2
        assert rect.y1 <= rect.y2

        # Could not make it work correctly # FIXME
        # txt = page.get_text_for_area(rect)
        # attrs = page.get_text_attributes_for_area(rect)

        r = Poppler.Rectangle()
        r.x1 = r.y1 = 1e10
        r.x2 = r.y2 = -1e10
        chars = []
        for k, l in enumerate(self.layout):
            if self.inside(l, rect, pad=0):
                self.rexpand(r, l, pad=0)
                chars.append(self.text[k])
                self.table_chars.add(k)
        txt = "".join(chars)

        # txt = page.get_text_for_area(r) # FIXME

        return txt, r

    def get_rest_text(self, bbox=None):
        """Returns the rest of a text, that is not
        recognized as data of a table.
        """
        # (cl,cr,ct,cb) == bbox if not None
        chars = []
        if bbox is not None:
            _ = bbrect = Poppler.Rectangle()
            _.x1, _.x2, _.y1, _.y2 = bbox
            assert _.x1 <= _.x2
            assert _.y1 <= _.y2
        for i, char in enumerate(self.text):
            if not i in self.table_chars:  # FIXME add inside bbox removal.
                l = self.layout[i]
                if bbox is not None and self.inside(l, bbrect):
                    continue
                chars.append(char)
        return "".join(chars)

    def get_rectangles_for_page(self, page):
        """Return all rectangles for all letters in the page..
        Used for debugging.

        Arguments:
        - `page`: referece to page
        """
        layout = self.layout
        if layout is None:
            raise RuntimeError("page is not chosen")

        answer = [(r.x1, r.y1, r.x2, r.y2) for r in layout]
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


class Extractor(object):
    """Extracts PDF content as whole or
    page by page.
    """

    def __init__(self,
                 infile,
                 pgs=None,
                 startpage=None,
                 endpage=None,
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
                 encoding="utf8",
                 rest_text=True,    # Return the rest of text as second parameter):
                 notify=None,
                 imsave=None,
                 page_layout=True,  # Try to follow page layout from top to bottom.
                 ):
        debug = False
        self.infile = infile
        self.pgs = pgs
        self.startpage = startpage
        self.endpage = endpage
        self.outfilename = outfilename
        self.greyscale_threshold = greyscale_threshold
        self.page = page
        self.crop = crop
        self.line_length = line_length
        self.bitmap_resolution = bitmap_resolution
        self.name = name
        self.pad = pad
        self.white = white
        self.black = black
        self.bitmap = bitmap
        if checkall:
            checkcrop = True
            checklines = True
            checkdivs = True
            checkcells = True
            checkletters = True
        self.checkcrop = checkcrop
        self.checklines = checklines
        self.checkdivs = checkdivs
        self.checkcells = checkcells
        self.checkletters = checkletters
        self.checkall = checkall
        checkany = False
        if checkcrop or checklines or checkdivs or checkcells or checkletters:
            debug = True
            checkany = True
        self.whitespace = whitespace
        self.boxes = boxes
        self.encoding = encoding
        self.rest_text = rest_text
        self.notify = notify
        self.imsave = imsave

        if imsave is not None:
            debug = True

        if debug:
            if imsave is None:
                raise RuntimeError("did not set image saving function")
            if not checkany:
                raise RuntimeError("did not set what pictures to be shown")

        self.debug = debug
        self.checkany = checkany

        self.outfile = outfilename if outfilename else "output"

        if pgs is not None:
            pgs, self.frow, self.lrow = (
                list(map(int, (str(pgs).split(":")))) + [None, None])[0:3]
            self.pgs = range(pgs, pgs + 1)
        else:
            self.set_pages(startpage, endpage)
        self.edoc = self.etree = None
        self.page_layout = page_layout

    def set_pages(self, startpage, endpage=None):
        if startpage is None:
            self.pgs = None
            return
        elif endpage is None:
            endpage = startpage
        self.pgs = range(startpage, endpage + 1)
        self.frow = self.lrow = None

    def initialize(self):
        """Initializes internal structures.
        """
        if self.pgs is None:
            raise ValueError("pages to be processed are not set")
        pdfdoc = self.pdfdoc = PopplerProcessor(self.infile)
        pdfdoc.resolution = self.bitmap_resolution
        pdfdoc.greyscale_threshold = self.greyscale_threshold
        self.pages = collections.OrderedDict()
        self.edoc = etree.Element("document")
        self.etree = etree.ElementTree(self.edoc)

    def process(self, notify=None):
        """Process the PDF file sending page number to
        `notify` function.
        """
        self.initialize()
        for page in self.pgs:
            if notify is not None:
                notify(page)
            if self.notify is not None:
                self.notify(page)
            self.process_page(page)

    def process_page(self, pg):

        pdfdoc = self.pdfdoc

        data, page = pdfdoc.get_image(pg - 1)  # Page numbers are 0-based.

        curr_page = self.pages[pg] = etree.SubElement(
            self.edoc, "page", number=str(pg))

        #-----------------------------------------------------------------------
        # image load section.

        height, width = data.shape[:
                                   2]  # If not to reduce to gray, the shape will be (,,3) or (,,4).

        pad = int(self.pad)
        height += pad * 2
        width += pad * 2

        # reimbed image with a white pad.
        bmp = ones((height, width), dtype=bool)

        thr = int(255.0 * self.greyscale_threshold / 100.0)

        bmp[pad:height - pad, pad:width - pad] = (data[:, :] > thr)

        checkany = self.checkany
        scale_to_pdf = self.bitmap_resolution / 72.

        if checkany:
            import random
            outfile = self.outfile + "-{:04d}".format(pg)
            # Set up Debuging image.
            img = zeros((height, width, 3), dtype=uint8)

            # img[:, :, :] = bmp * 255 # In case of colored input image

            img[:, :, 0] = bmp * 255
            img[:, :, 1] = bmp * 255
            img[:, :, 2] = bmp * 255

            if self.checkdivs or self.checkcells or self.checkletters:
                imgfloat = img.astype(float)

            if self.checkletters:  # Show bounding boxes for each text object.
                img = (imgfloat / 2.).astype(uint8)
                rectangles = pdfdoc.get_rectangles_for_page(pg)
                lrn = len(rectangles)
                for k, r in enumerate(rectangles):
                    x1, y1, x2, y2 = [
                        int(scale_to_pdf * float(k)) + pad for k in r
                    ]
                    img[y1:y2, x1:x2] += col(random.random()).astype(uint8)
                self.imsave(outfile + "-letters.png", img)

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

        scaled = [(_v - pad) * scale_to_pdf for _v in [l, t, r, b]]
        [curr_page.set("bbox-" + _k, str(_v))
         for _k, _v in zip(["left", "top", "right", "bottom"], scaled)]
        (sl, st, sr, sb) = scaled
        curr_page.set("bbox-width", str(abs(sr - sl)))
        curr_page.set("bbox-height", str(abs(st - sb)))
        curr_page.set("bounding-box", " ".join(map(str, [sl, st, sr, sb])))
        [curr_page.set(_k, str(_v))
         for _k, _v in zip(["width", "height"], page.get_size())]

        # Mark bounding box.
        # bmp[t, :] = False
        # bmp[b, :] = False
        # bmp[:, l] = False
        # bmp[:, r] = False

        def boxOfString(x, p):
            s = x.split(":")
            if len(s) < 4:
                raise ValueError(
                    "boxes have format left:top:right:bottom[:page]")
            return ([self.bitmap_resolution * float(x) + pad for x in s[0:4]] +
                    [p if len(s) < 5 else int(s[4])])

        if checkany:
            print("Bounding box(l,b,r,t): {}".format((l, b, r, t)))

    # translate crop to paint white.

        whites = []
        # curr_page.whites = whites
        if self.crop:
            (l, t, r, b, p) = boxOfString(self.crop, pg)
            whites.extend([(0, 0, l, height, p), (0, 0, width, t, p),
                           (r, 0, width, height, p), (0, b, width, height, p)])

    # paint white ...
        if self.white:
            whites.extend([boxOfString(b, pg) for b in white])

        for (l, t, r, b, p) in whites:
            if p == pg:
                bmp[t:b + 1, l:r + 1] = 1
                if checkany:
                    img[t:b + 1, l:r + 1] = [255, 255, 255]

    # paint black ...
        if self.black:
            for b in self.black:
                (l, t, r, b) = [self.bitmap_resolution * float(x) + pad
                                for x in b.split(":")]
                bmp[t:b + 1, l:r + 1] = 0
                if checkany:
                    img[t:b + 1, l:r + 1] = [0, 0, 0]

        if self.checkcrop:
            self.imsave(outfile + "-crop.png", img)

    #-----------------------------------------------------------------------
    # Line finding section.
    #
    # Find all vertical or horizontal lines that are more than lthresh
    # long, these are considered lines on the table grid.

        lthresh = int(self.line_length * self.bitmap_resolution)
        vs = zeros(width, dtype=uint8)

        for i in range(l, r + 1):
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

        for j in range(t, b + 1):
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

        if self.checklines:
            for i in vd:
                img[:, i] = [255, 0, 0]  # red

            for j in hd:
                img[j, :] = [0, 0, 255]  # blue
            self.imsave(outfile + "-lines.png", img)

            #-----------------------------------------------------------------------
            # divider checking.
            #
            # at this point vd holds the x coordinate of vertical  and
            # hd holds the y coordinate of horizontal divider tansitions for each
            # vertical and horizontal lines in the table grid.

        def isDiv(a, l, r, t, b):
            # if any col or row (in axis) is all zeros ...
            return sum(sum(bmp[t:b, l:r], axis=a) == 0) > 0

        if self.checkdivs:
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
            self.imsave(outfile + "-divs.png", img)

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
                            bot |= isDiv(1, vd[2 * (i + k) - 1], vd[2 *
                                                                    (i + k)],
                                         hd[2 * (j + v)], hd[2 * (j + v) + 1])
                        if not bot:
                            v = v + 1
                    cells.append((i, j, u, v))
                    touched[j:j + v, i:i + u] = True
                i = i + 1
            j = j + 1

        if self.checkcells:
            nc = len(cells) + 0.
            img = (imgfloat / 2.).astype(uint8)
            for k in range(len(cells)):
                (i, j, u, v) = cells[k]
                (l, r, t, b) = (vd[2 * i + 1], vd[2 * (i + u)], hd[2 * j + 1],
                                hd[2 * (j + v)])
                img[t:b, l:r] += col(k * 0.9 / nc + 0.1 * random.random(
                )).astype(uint8)

            self.imsave(outfile + "-cells.png", img)

            #-----------------------------------------------------------------------
            # fork out to extract text for each cell.

        def getCell(_coordinate):
            (i, j, u, v) = _coordinate
            (l, r, t, b) = (vd[2 * i + 1], vd[2 * (i + u)], hd[2 * j + 1],
                            hd[2 * (j + v)])
            ret, rect = pdfdoc.get_text(page, l - pad, t - pad, r - l, b - t)

            if self.checkletters:
                (x1, y1, x2, y2) = [
                    int(scale_to_pdf * float(rrr) + pad)
                    for rrr in [rect.x1, rect.y1, rect.x2, rect.y2]
                ]
                img[y1:y2, x1:x2] += col(random.random()).astype(uint8)

            return (i, j, u, v, pg, ret)

        if self.checkletters:
            img = (imgfloat / 2.).astype(uint8)

        lrow, frow = self.lrow, self.frow
        if self.boxes:
            cells = [x + (pg,
                          "", ) for x in cells
                     if (frow is None or (x[1] >= frow and x[1] <= lrow))]
        else:
            cells = [getCell(x) for x in cells
                     if (frow is None or (x[1] >= frow and x[1] <= lrow))]

        if self.checkletters:
            self.imsave(outfile + "-text-locations.png", img)
        self._cells = cells
        text = self._text = None

        def cell_proc(tbl, cl):
            c = etree.SubElement(tbl, "cell")
            [c.set(k, str(v)) for k, v in zip("xywhp", map(str, cl))]
            if cl[-1]:
                c.text = cl[-1]
            (i, j, u, v) = cl[:4]
            (cl, cr, ct, cb) = (vd[2 * i + 1], vd[2 * (i + u)], hd[2 * j + 1],
                                hd[2 * (j + v)])
            (cl, cr, ct,
             cb) = [(_v - pad) * scale_to_pdf for _v in (cl, cr, ct, cb)]
            (cw, ch) = cr - cl, abs(ct - cb)
            [c.set("bbox-" + k, str(v))
             for k, v in zip(["left", "right", "top", "bottom", "width",
                              "height"], map(str, (cl, cr, ct, cb, cw, ch)))]
            return cl, cr, ct, cb, i, j, u, v

        bbox = None
        table = None
        if cells:
            table = etree.Element("table")
            mat = numpy.array([cell_proc(table, cl) for cl in cells])
            cl, cr, ct, cb = min(mat[:, 0]), max(mat[:, 1]), min(
                mat[:, 2]), max(mat[:, 3])
            _w, _h = max(mat[:, 4] + mat[:, 6]) - 1, max(mat[:, 5] +
                                                         mat[:, 7]) - 1
            cw = cr - cl
            ch = cb - ct
            [table.set("bbox-" + _k, str(_v))
             for _k, _v in zip(["left", "right", "top", "bottom", "width",
                                "height"], [cl, cr, ct, cb, cw, ch])]
            table.set("width", str(int(_w)))
            table.set("height", str(int(_h)))
            table.set("page", str(pg))
            bbox = (cl, cr, ct, cb)

        etext = None
        if self.rest_text:
            if self.page_layout:
                self.follow_layout(page, curr_page, table)
            else:
                if table is not None:
                    curr_page.append(table)
                text = pdfdoc.get_rest_text(bbox=bbox)

                if text is not None:
                    etext = etree.Element("text")
                    etext.text = text
                if etext is not None:
                    curr_page.append(etext)

    def follow_layout(self, page, curr_page, table=None):
        # FIXME Implemented a very simple method
        # just enumerating chars making string with them
        # till \n will found

        bbox = None
        if table is not None:
            bbox = l, t, r, b = [float(table.get("bbox-" + k))
                                 for k in ["left", "top", "right", "bottom"]]
        if bbox is not None:
            _ = bbrect = Poppler.Rectangle()
            _.x1, _.y1, _.x2, _.y2 = bbox
            assert _.x1 <= _.x2
            assert _.y1 <= _.y2

        class Context(object):
            pass

        ctx = Context()
        ctx.text = etree.Element("text")
        ctx.style = None
        ctx.chars = []
        l, t, r, b = str_bb = (1e10, 1e10, -1e10, -1e10)
        ctx.line = etree.Element("line")

        ctx.attributes = page.get_text_attributes()
        ctx.lattrs = len(ctx.attributes)
        ctx.aidx = 0
        ctx.attr_dict = {}

        def store(ctx, endline=False):
            style = etree.SubElement(ctx.line, "style")
            style.text = "".join(ctx.chars)
            [style.set(k, str(v))
             for k, v in ctx.attr_dict.items()]  # .attrib.update(.)? FIXME
            h = b - t
            w = r - l
            if endline:
                [ctx.line.set("bbox-" + k, str(v))
                 for k, v in zip(["left", "top", "right", "bottom", "width",
                                  "height"], [l, t, r, b, w, h])]
                if len(ctx.line) > 0:
                    ctx.text.append(ctx.line)
                    ctx.line = etree.Element("line")

        def get_attrs(i, ctx):
            # global aidx, lattrs, attributes, attr_dict
            answer = {}
            step = False
            while True:
                if ctx.aidx >= ctx.lattrs:
                    raise RuntimeError("wrong sequence")
                attr = ctx.attributes[ctx.aidx]
                if i >= attr.start_index and i <= attr.end_index:
                    if not step:
                        return step, ctx.attr_dict
                    _ = color = attr.color
                    color = _.red, _.green, _.blue
                    color = "{:f} {:f} {:f}".format(*color)
                    answer["color"] = color
                    answer["underline"] = "1" if attr.is_underlined else "0"
                    answer["font-spec"] = attr.font_name
                    aname = attr.font_name.split(",", 1)
                    if len(aname) >= 2:
                        font_name, modifiers = aname
                    else:
                        font_name = aname[0]
                        modifiers = ""
                    answer["font-name"] = font_name
                    modifiers = modifiers.lower().split(",")
                    answer["bold"] = "1" if "bold" in modifiers else "0"
                    answer["italic"] = "1" if "italic" in modifiers else "0"
                    answer["modifiers"] = " ".join(modifiers)
                    answer["font-size"] = str(attr.font_size)
                    ctx.attr_dict.clear()
                    ctx.attr_dict.update(answer)
                    return step, ctx.attr_dict
                else:
                    step = True
                    ctx.aidx += 1
            assert False

        for i, _ in enumerate(zip(self.pdfdoc.layout, self.pdfdoc.text)):
            la, c = _
            step, _ = get_attrs(i, ctx)
            if i in self.pdfdoc.table_chars:  # the character is already in a table.
                if table is None:
                    continue
                else:
                    if len(ctx.text) > 0:
                        curr_page.append(ctx.text)
                        ctx.text = etree.Element("text")
                        ctx.line = etree.Element("line")
                    curr_page.append(table)
                    table = None
                    ctx.style = None
            if bbox is not None and self.pdfdoc.inside(la, bbrect):
                continue
            ctx.chars.append(c)
            if l > la.x1:
                l = la.x1
            if t > la.y1:
                t = la.y1
            if r < la.x2:
                r = la.x2
            if b < la.y2:
                b = la.y2
            endline = c == "\n"
            if endline or step:
                store(ctx, endline=endline)
                ctx.chars = []
                if endline:
                    l, t, r, b = (1e10, 1e10, -1e10, -1e10)
                    ctx.style = None
        if len(ctx.chars) > 0:
            store(ctx, endline=True)
        if len(ctx.text) > 0:
            curr_page.append(ctx.text)

    def xml_write(self, f, pretty_print=True, encoding="UTF-8"):
        self.etree.write(f, pretty_print=pretty_print, encoding=encoding)

    def as_xhtml_tree(self, text="div", line="div"):
        def update(element, attrib, include=None):
            for k, v in attrib.items():
                if include is not None:
                    if k in include:
                        element.set("data-pdf-" + k, v)
                else:
                    element.set("data-pdf-" + k, v)

        tree = copy.deepcopy(self.etree)
        for e in tree.getroot().getiterator():
            attrib = {}
            attrib.update(e.attrib)
            e.attrib.clear()
            if e.tag == "document":
                e.tag = "div"
                e.set("class", "pdf-document")
            elif e.tag == "text":
                e.tag = text
                e.set("class", "pdf-text")
                page = attrib.get("page", None)
                if page is not None:
                    e.set("data-pdf-page-number", page)
            elif e.tag == "line":
                e.tag = line
                e.set("class", "pdf-line")
                update(e, attrib)
            elif e.tag == "page":
                e.tag = "div"
                e.set("data-pdf-page-number", attrib.get("number"))
                e.set("class", "pdf-page")
            elif e.tag == "style":
                e.tag = "span"
                etext = e.text
                b = attrib.get("bold", "0")
                i = attrib.get("italic", "0")
                b, i = map(int, [b, i])
                if b or i:
                    prev = e
                    if b:
                        prev.text = None
                        b = etree.SubElement(prev, "b")
                        b.text = etext
                        prev = b
                    if i:
                        prev.text = None
                        i = etree.SubElement(prev, "i")
                        i.text = etext
                update(e, attrib,
                       ["font-name", "font-size", "color", "underline"])
            elif e.tag == "table":
                htable = self._as_xhtml_table(e)
                _p = e.getparent()
                e.addprevious(htable)
                _p.remove(e)
                update(htable, attrib)
            else:
                update(e, attrib)
        return tree

    def _as_xhtml_table(self, table, chtml=False):
        nc = len(table)
        if nc == 0:
            return None
        htable = etree.Element("table")
        if chtml:
            htable.set("border", "1")
            htable.set("cellspacing", "0")
            htable.set("style", "border-spacing:0")
            tr = None
        oj = -1
        for cell in table.iterchildren():
            (i, j, u, v, pg) = [int(cell.get(v)) for v in "xywhp"]
            value = cell.text if cell.text is not None else ""

            if j > oj:
                tr = etree.SubElement(htable, "tr")
                oj = j
            td = etree.SubElement(tr, "td")
            td.text = value
            if u > 1:
                td.set("colspan", str(u))
            if v > 1:
                td.set("rowspan", str(v))
            if chtml:
                td.set("style", "background-color: #%02x%02x%02x" %
                       tuple(128 + col(k / (nc + 0.))))
        return htable

    def xhtml_write(self, f, pretty_print=True, encoding="UTF-8"):
        tree = self.as_xhtml_tree()
        tree.write(f, pretty_print=pretty_print, encoding=encoding)

    def reduce(self, remove_pages=False, join_styles=True, inplace=False):
        """Reduces structure of etree,
        joining styles, removing pages, etc.
        """
        tree = copy.deepcopy(self.etree)
        edoc = tree.getroot()
        if remove_pages:
            pages = edoc.iterfind("page")
            for p in pages:
                for c in p.iterchildren():
                    p.remove(c)
                    if c.tag == "text":
                        c = self._join_styles(c, join_styles)
                    edoc.append(c)
                    c.set("page", p.get("number"))
                edoc.remove(p)
        else:
            pages = edoc.iterfind("page")
            for p in pages:
                for c in p.iterchildren("text"):
                    self._join_styles(c, join_styles)
        if inplace:
            self.etree = tree
        return tree

    def _join_styles(self, text, really):
        """Join adjacent style tags.
        """
        if not really:
            return text
        for line in text.iterfind("line"):
            s = None
            for style in line.iterfind("style"):
                if s is None:
                    s = style
                    if s.text is None:
                        s.text = ''
                    continue
                if s.attrib == style.attrib:
                    s.text += style.text
                    line.remove(style)
                    continue
                if not s.text:
                    line.remove(s)
                    s = None
        return text

    def _get_cells(self, pg):
        tbls = self.pages[pg].iterfind("table")
        cells = []
        for tbl in tbls:
            cls = []
            for cl in tbl.iterfind("cell"):
                l = [int(cl.get(v)) for v in "xywhp"]
                if cl.text:
                    l.append(cl.text)
                else:
                    l.append('')
                cls.append(l)
            cells.append(cls)
        if len(cells) == 1:
            cells = cells[0]
        return cells

    def cells(self, pg=None):
        """Return all the cells found.
        """
        if pg is not None:
            return self._get_cells(pg)
        cells = []
        for p in self.pages.keys():
            cells.extend(self._get_cells(p))
        return cells

    def texts(self, pg=None):
        """Return all the cells found.
        """
        q = "text/line/style"

        if pg is None:
            node = self.pages[pg]
        else:
            q = ".//" + q
            node = self.edoc

        styles = node.iterfind(q)
        text = ""
        t = etree.Element("text")
        for s in styles:
            b = i = False
            b = s.get("bold", "0") == "1"
            i = s.get("italic", "0") == "1"
            t = s.text
            if i:
                t = "<i>" + t + "</i>"
            if b:
                t = "<b>" + t + "</b>"
            text += t
        # text = "".join([_e.text for _e in etexts])
        return text

    def output(self,
               pgs=None,
               cells_csv_filename=None,
               cells_json_filename=None,
               cells_xml_filename=None,
               table_csv_filename=None,
               table_html_filename=None,
               table_list_filename=None,
               name=None,
               output_type=None):
        """Output recognition result in various
        formats defined by parameters.
        """
        for pg, page in self.pages.items():
            cells = self.cells(pg)
            text = self.texts(pg)
            pref = "page-{:04d}".format(pg)
            output(
                cells,
                text=text,
                pgs=None,
                prefix=pref,
                cells_csv_filename=cells_csv_filename,
                cells_json_filename=cells_json_filename,
                cells_xml_filename=cells_xml_filename,
                table_csv_filename=table_csv_filename,
                table_html_filename=table_html_filename,
                table_list_filename=table_list_filename,
                infile=self.infile,
                name=name,
                output_type=output_type)

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
           output_type=None,
           text=None,
           prefix=None):

    output_types = [
        dict(
            filename=cells_csv_filename, function=o_cells_csv), dict(
                filename=cells_json_filename, function=o_cells_json), dict(
                    filename=cells_xml_filename, function=o_cells_xml), dict(
                        filename=table_csv_filename, function=o_table_csv),
        dict(
            filename=table_html_filename, function=o_table_html), dict(
                filename=table_list_filename, function=o_table_list)
    ]

    for entry in output_types:
        if entry["filename"]:
            if entry["filename"] != sys.stdout:
                filename = entry["filename"]
                if "{}" in filename:
                    filename = filename.format(prefix)
                outfile = open(filename, 'wb')
            else:
                outfile = sys.stdout

            entry["function"](cells,
                              pgs,
                              outfile=outfile,
                              name=name,
                              infile=infile,
                              output_type=output_type,
                              text=text)

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
                output_type=None,
                text=None):
    """Output XML formatted cell data"""
    outfile = outfile or sys.stdout
    #defaults
    infile = infile or ""
    name = name or ""

    def _lambda(a):
        return x.set(*a)

    table = etree.Element("table")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    text = None  # FIXME We will ignore text in cell_xml mode
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if text is not None:
        root = etree.Element("document")
        if isinstance(text, str):
            t = etree.SubElement(root, "text")
            t.text = text
        else:
            t = text
        root.append(table)
        doc = etree.ElementTree(root)
    else:
        doc = etree.ElementTree(table)
        root = table
    if infile:
        table.set("src", infile)
    if name:
        table.set("name", name)
    for cl in cells:
        x = etree.SubElement(table, "cell")
        map(_lambda, zip("xywhp", map(str, cl)))
        if cl[5] != "":
            x.text = cl[5]
    doc.write(outfile, pretty_print=True, encoding="UTF-8")


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
                 text=None,
                 infile=None):
    """Output HTML formatted table"""

    oj = 0
    opg = 0
    doc = etree.Element("div")
    div = root = doc
    if text is not None:
        if isinstance(text, str) and text.strip():
            p = etree.SubElement(div, "p")
            p.set("align", "justify")
            text = text.rstrip()
            pars = text.split("\n")
            for par in pars:
                _div = etree.SubElement(p, "div")
                _div.set("class", "sentence")
                _div.text = par
        elif isinstance(text, list):
            raise RuntimeError("not implemented")

    nc = len(cells)
    if nc > 0:
        table = etree.SubElement(div, "table")
        if (output_type == "table_chtml"):
            table.set("border", "1")
            table.set("cellspacing", "0")
            table.set("style", "border-spacing:0")
        tr = None
        for k in range(nc):
            (i, j, u, v, pg, value) = cells[k]
            if j > oj or pg > opg:
                if pg > opg:
                    s = "Name: " + name + ", " if name else ""
                    table.append(
                        etree.Comment(s + ("Source: %s page %d." % (infile, pg)
                                           )))
                #if tr:
                #    table.appendChild(tr)
                #tr = doc.createElement("tr")
                tr = etree.SubElement(table, "tr")
                oj = j
                opg = pg
            td = etree.SubElement(tr, "td")
            if value != "":
                td.text = value
            if u > 1:
                td.set("colspan", str(u))
            if v > 1:
                td.set("rowspan", str(v))
            if output_type == "table_chtml":
                td.set("style", "background-color: #%02x%02x%02x" %
                       tuple(128 + col(k / (nc + 0.))))
    outfile.write(
        etree.tostring(
            doc, method="html", pretty_print=True, encoding="UTF-8"))


def process_page(infile, pgs, **kwargs):
    """Performs extraction. It is API function of the
    previous version of the library.
    """
    ext = Extractor(infile=infile, pgs=pgs, **kwargs)

    ext.process()

    return ext.cells()

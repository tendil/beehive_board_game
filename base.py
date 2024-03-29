import copy
import math
from PIL import Image, ImageDraw


# DO NOT MODIFY THIS FILE
# THIS FILE IS NOT UPLOADED TO BRUTE (all changes in it will be ignored by Brute)


class Board:
    def __init__(self, myIsUpper, size, myPieces, rivalPieces):
        self.size = size  # integer, size of the board
        self.myMove = 0  # integer, index of actual move
        self.board = {}  # dict of board, use self.board[p][q] to acess cell (p,q)
        self.myColorIsUpper = myIsUpper  # if true, by figures are big (A,Q,S,G,B), otherwise there are small (a,q,s,g,b)
        self.algorithmName = "some algorithm"
        self.playerName = "some name"
        self.tournament = 0  # filled by Brute, if True, player is run in tournament mode

        self.myPieces = myPieces.copy()  # dict of key=animal, value = number of available, self.myPieces["b"] = 3
        self._myPiecesOriginal = myPieces.copy()

        self.rivalPieces = rivalPieces.copy()
        self._rivalPiecesOriginal = rivalPieces.copy()

        # the rest of the coe is just for drawing to png

        self._images = {}
        self._imagesSmall = {}

        for imagename in ["ant", "beetle", "bee", "spider", "grasshopper"]:
            self._images[imagename] = Image.open("images/{}.png".format(imagename)).resize((70, 70))
            self._imagesSmall[imagename] = Image.open("images/{}.png".format(imagename)).resize((20, 20))

        # create empty board as a dictionary
        for p in range(-self.size, self.size):
            for q in range(-self.size, self.size):
                if self.inBoard(p, q):
                    if not p in self.board:
                        self.board[p] = {}
                    self.board[p][q] = ""

        # this is for visualization and to synchronize colors between png/js
        self._colors = {}
        self._colors[-1] = "#fdca40"  # sunglow
        self._colors[0] = "#ffffff"  # white
        self._colors[1] = "#947bd3"  # medium purple
        self._colors[2] = "#ff0000"  # red
        self._colors[3] = "#00ff00"  # green
        self._colors[4] = "#0000ff"  # blue
        self._colors[5] = "#566246"  # ebony
        self._colors[6] = "#a7c4c2"  # opan
        self._colors[7] = "#ADACB5"  # silver metalic
        self._colors[8] = "#8C705F"  # liver chestnut
        self._colors[9] = "#FA7921"  # pumpkin
        self._colors[10] = "#566E3D"  # dark olive green

    def inBoard(self, p, q):
        """ return True if (p,q) is valid coordinate """
        return (q >= 0) and (q < self.size) and (p >= -(q // 2)) and (p < (self.size - q // 2))

    def rotateRight(self, p, q):
        pp = -q
        qq = p + q
        return pp, qq

    def rotateLeft(self, p, q):
        pp = p + q
        qq = -p
        return pp, qq

    def letter2image(self, lastLetter):
        impaste = None
        impaste2 = None
        lastLetter = lastLetter.lower()
        if lastLetter == "q":
            impaste, impaste2 = self._images["bee"], self._imagesSmall["bee"]
        elif lastLetter == "b":
            impaste, impaste2 = self._images["beetle"], self._imagesSmall["beetle"]
        elif lastLetter == "s":
            impaste, impaste2 = self._images["spider"], self._imagesSmall["spider"]
        elif lastLetter == "g":
            impaste, impaste2 = self._images["grasshopper"], self._imagesSmall["grasshopper"]
        elif lastLetter == "a":
            impaste, impaste2 = self._images["ant"], self._imagesSmall["ant"]
        return impaste, impaste2

    def saveImage(self, filename, HL={}, LINES=[], HLA={}):
        """ draw actual board to png. Empty cells are white, -1 = red, 1 = green, other values according to
            this list
            -1 red, 0 = white, 1 = green
            HL is dict of coordinates and colors, e.g.
            HL[(3,4)] = #RRGGBB #will use color #RRGGBB to highlight cell (3,4)
            LINES is list of extra lines to be drawn in format
            LINES = [ line1, line2 ,.... ], where each line is [#RRGGBB, p1, q1, p2,q2] - will draw line from cell (p1,q1) to cell (p2,q2)
        """

        def pq2hexa(p, q):
            cx = cellRadius * (math.sqrt(3) * p + math.sqrt(3) / 2 * q) + cellRadius
            cy = cellRadius * (0 * p + 3 / 2 * q) + cellRadius

            pts = []
            for a in [30, 90, 150, 210, 270, 330]:
                nx = cx + cellRadius * math.cos(a * math.pi / 180)
                ny = cy + cellRadius * math.sin(a * math.pi / 180)
                pts.append(nx)
                pts.append(ny)
            return cx, cy, pts

        def drawPieces(piecesToDraw, piecesToDrawOriginal, draw, p, q, HLA={}):
            # HLA is dict, key is animal for highlight, value is color, e.g. HLA["a"] = "#RRGGBB" will highlight my own piece 'a'
            for animal in piecesToDraw:
                for v in range(piecesToDrawOriginal[animal]):
                    # draw this animal
                    cx, cy, pts = pq2hexa(p, q)
                    color = "#ff00ff"

                    lastLetter = animal
                    if lastLetter.islower():
                        color = self._colors[-1]
                    else:
                        color = self._colors[1]
                    if v < piecesToDraw[animal] and animal in HLA:
                        color = HLA[animal]

                    draw.polygon(pts, fill=color)
                    pts.append(pts[0])
                    pts.append(pts[1])
                    draw.line(pts, fill="black", width=1)

                    lastLetter = animal.lower()
                    icx = int(cx) - cellRadius // 1;
                    icy = int(cy) - cellRadius // 1;
                    if v < piecesToDraw[animal]:
                        impaste, impaste2 = self.letter2image(lastLetter)
                        if impaste:
                            img.paste(impaste, (int(icx), int(icy)), impaste)
                    p += 1

        cellRadius = 35
        cellWidth = int(cellRadius * (3 ** 0.5))
        cellHeight = 2 * cellRadius

        width = cellWidth * self.size + cellRadius * 3
        height = cellHeight * self.size

        img = Image.new('RGB', (width, height), "white")

        draw = ImageDraw.Draw(img)

        lineColor = (50, 50, 50)

        allQ = []
        allP = []

        for p in self.board:
            allP.append(p)
            for q in self.board[p]:
                allQ.append(q)
                cx, cy, pts = pq2hexa(p, q)

                color = "#ff00ff"  # pink is for values out of range -1,..10
                if self.isEmpty(p, q, self.board):
                    color = self._colors[0]
                else:
                    lastLetter = self.board[p][q][-1]
                    if lastLetter.islower():
                        color = self._colors[-1]
                    else:
                        color = self._colors[1]
                #                    if lastLetter.lower() in "bB" and len(self.board[p][q]) > 1:
                #                        color = self._colors[2] #red beetle

                if (p, q) in HL:
                    color = HL[(p, q)]
                draw.polygon(pts, fill=color)

                if not self.isEmpty(p, q, self.board) and self.board[p][q][-1].lower() in "bB" and len(
                        self.board[p][q]) > 1:
                    # draw half of the polygon in red color to highlight that beetle is on the top
                    polygon2 = pts[6:] + pts[:2]
                    draw.polygon(polygon2, fill=self._colors[2])

                pts.append(pts[0])
                pts.append(pts[1])
                draw.line(pts, fill="black", width=1)
                draw.text([cx - 3, cy - 3], "{} {}".format(p, q), fill="black", anchor="mm")
                if not self.isEmpty(p, q, self.board):
                    draw.text([cx, cy], "{}".format(self.board[p][q]), fill="black", anchor="mm")
                    lastLetter = self.board[p][q][-1].lower()

                    icx = int(cx) - cellRadius // 1;
                    icy = int(cy) - cellRadius // 1;
                    impaste, impaste2 = self.letter2image(lastLetter)

                    if impaste:
                        img.paste(impaste, (int(icx), int(icy)), impaste)

        maxq = max(allQ)
        minp = min(allP)
        maxq += 2
        minp += 1

        drawPieces(self.myPieces, self._myPiecesOriginal, draw, minp, maxq, HLA)
        maxq += 1
        drawPieces(self.rivalPieces, self._rivalPiecesOriginal, draw, minp, maxq, HLA)

        for line in LINES:
            color, p1, q1, p2, q2 = line
            cx1, cy1, _ = pq2hexa(p1, q1)
            cx2, cy2, _ = pq2hexa(p2, q2)
            draw.line([cx1, cy1, cx2, cy2], fill=color, width=2)

        img.save(filename)

    def print(self, board):
        for p in board:
            for q in board[p]:
                value = board[p][q]
                if value == "":
                    value = ".."
                print(value, end="  ")
            print()

    def isMyColor(self, p, q, board):
        """ assuming board[p][q] is not empty """
        return ((not self.myColorIsUpper) and board[p][q][-1].islower()) or (
                    self.myColorIsUpper and board[p][q][-1].isupper())

    def isEmpty(self, p, q, board):
        return board[p][q] == ""

    def a2c(self, p, q):
        x = p
        z = q
        y = -x - z
        return x, y, z

    def c2a(self, x, y, z):
        p = x
        q = z
        return p, q

    def distance(self, p1, q1, p2, q2):
        """ return distance between two cells (p1,q1) and (p2,q2) """
        x1, y1, z1 = self.a2c(p1, q1)
        x2, y2, z2 = self.a2c(p2, q2)
        dist = (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2
        return dist

from homoGraphs.homoGraph import HomoGraph
from homoGraphs.shapeGraph import ShapeGraph


class DetectorModel():
    def __init__(self, homoGraph_path):
        is_shape = "shape" in homoGraph_path
        if is_shape:
            shapeGraph = ShapeGraph(homoGraph_path)
            self.homoGraph = shapeGraph
        else:
            homoGraph = HomoGraph(homoGraph_path)
            self.homoGraph = homoGraph

    def detectLexicons(self, text: str):
        N = len(text)
        res = []
        for i in range(N):
            for lexicon in self.homoGraph.lexicons:
                if len(lexicon) > N - i:
                    continue
                flag = True
                for j in range(len(lexicon)):
                    if not self.homoGraph.neighborsHas(text[i+j], lexicon[j]):
                        flag = False
                        break
                if flag:
                    res.append({"pos": i, "lexicon": lexicon})
        return res

if __name__ == "__main__":
    pass

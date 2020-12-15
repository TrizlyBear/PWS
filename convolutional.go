package cnn

import (
	"errors"
	//"fmt"
	//"math"
)

type Conv2D struct {
	Ksize  int
	Stride int
}

func (e Conv2D) Forward(in [][]float64, kfilter []float64) ([][]float64, error) {
	if len(in[0]) != len(in) {
		return [][]float64{{-1}}, errors.New("Input is not a square")
	}
	var outsize = (len(in)-e.Ksize)/e.Stride + 1
	var out = make([][]float64, outsize)
	//fmt.Printf("outsize %v", outsize)
	for y, _ := range in {
		for x, _ := range in[y] {
			if y%(e.Stride) == 0 && x%(e.Stride) == 0 && x+1 < len(in) && y+1 < len(in[x]) {
				var all = make([]float64, 0)
				var q = 1
				for q := q; q < e.Ksize+1; q++ {
					var w = 1
					for w := w; w < e.Ksize+1; w++ {
						all = append(all, in[q+y-1][w+x-1]*kfilter[len(all)])
					}
				}
				var av = Sum(all)
				out[y/e.Stride] = append(out[y/e.Stride], av)
			}
		}
	}
	return out, nil
}

package activation

import (
	"errors"
	"github.com/TrizlyBear/PWS/math"
)

// Maximum pooling layer
type MaxPooling struct {
	max [][][]struct{
		y int
		x int
	}
}

func (e *MaxPooling) Forward(in [][][]float64) [][][]float64{
	(*e).max = [][][]struct{
		y int
		x int
	}{}

	out := [][][]float64{}
	for l,_ := range in {
		lout := make([][]float64, len(in[l]) / 2)
		iout := make([][]struct{
			y int
			x int
		}, len(in[l]) / 2)
		for y := 0; y < len(in[l]) - 1; y += 2 {
			for x := 0; x < len(in[l][y]) - 1; x += 2 {
				o, ox, oy := math.MaxIndex([][]float64{in[l][y][x:x+2], in[l][y+1][x:x+2]})
				lout[y / 2] = append(lout[y / 2], o)
				iout[y / 2] = append(iout[y / 2], struct {
					y int
					x int
				}{oy + y, ox + x})

			}
		}
		(*e).max = append((*e).max, iout)
		out = append(out, lout)
	}
	return out
}

func (e *MaxPooling) Backward(error [][][]float64, lr float64) [][][]float64 {
	out := [][][]float64{}
	for l,_ := range error {
		lout := math.Zeros(len(error[l])*2,len(error[l][0])*2).([][]float64)
		for y,_ := range error[l] {
			for x,_ := range error[l][y] {
				lout[e.max[l][y][x].y][e.max[l][y][x].x]  += error[l][y][x]
			}
		}

		out = append(out, lout)
	}

	return out
}

// Maximum pooling forward function
/*
func (e MaxPooling) Forward(in [][]float64) ([][]float64, error) {
	if len(in[0]) != len(in) {
		return [][]float64{{-1}}, errors.New("Input is not a square")
	}
	var outsize = (len(in)-e.Ksize)/e.Stride + 1
	var out = make([][]float64, outsize)
	for y, _ := range in {
		for x, _ := range in[y] {
			if y%(e.Stride) == 0 && x%(e.Stride) == 0 && x+1 < len(in) && y+1 < len(in[x]) {
				var all = make([]float64, 0)
				var q = 1
				for q := q; q < e.Ksize+1; q++ {
					var w = 1
					for w := w; w < e.Ksize+1; w++ {
						all = append(all, in[q+y-1][w+x-1])
					}
				}
				var av = math.Max(all)
				out[y/e.Stride] = append(out[y/e.Stride], av)
			}
		}
	}
	return out, nil
}
 */

// Average pooling layer
type AvgPooling struct {
	Ksize  int
	Stride int
}

// Average pooling forward function
func (e AvgPooling) Forward(in [][]float64) ([][]float64, error) {
	if len(in[0]) != len(in) {
		return [][]float64{{-1}}, errors.New("Input is not a square")
	}
	var outsize = (len(in)-e.Ksize)/e.Stride + 1
	var out = make([][]float64, outsize)
	for y, _ := range in {
		for x, _ := range in[y] {
			if y%e.Stride == 0 && x%e.Stride == 0 {
				var all = make([]float64, e.Ksize^2)
				var q = 1
				for q := q; q < e.Ksize+1; q++ {
					var w = 1
					for w := w; w < e.Ksize+1; w++ {
						all = append(all, in[q+y-1][w+x-1])
					}
				}
				var av = math.Mean(all)
				out[(y)/e.Stride] = append(out[(y)/e.Stride], av)
			}
		}
	}
	return out, nil
}

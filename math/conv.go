package math

import (
	"errors"
)

func Convolve(in [][]float64, k [][]float64, s int) (out [][]float64, err error) {
	if (len(in) - len(k)) % s != 0 || (len(in[0]) - len(k[0])) % s != 0 {
		return nil, errors.New("Stride doesn't fit input with kernel")
	}

	var xP int = 0
	var yP int = 0

	for yP - 1 + s + len(k) <= len(in) {
		out = append(out, []float64{})
		for xP - 1 + s + len(k[0]) <= len(in[0]) {
			value := .0
			for y,_ := range k {
				for x,_ := range k[y] {
					value += in[yP + y][xP + x] * k[y][x]
				}
			}
			out[len(out)-1] = append(out[len(out)-1], value)
			xP += s
		}
		xP = 0
		yP += s
	}

	return out, nil
}

package layers

import (
	"github.com/TrizlyBear/PWS/math"
)

type Conv2D struct {
	KernelSize	struct{
		X int
		Y int
	}
	Depth		int
	Stride		int
	kernels 	[][][][]float64
	bias		[]float64
	insize		struct{
		x int
		y int
	}
	input		[][][]float64
	inputdepth	int
	init 		bool
}

func (e *Conv2D) Forward(in [][][]float64) (out [][][]float64) {

	if e.init != true {
		(*e).inputdepth = len(in)
		(*e).kernels = math.Rand(e.inputdepth, e.Depth, e.KernelSize.Y,e.KernelSize.X).([][][][]float64)
		(*e).bias = math.Rand(e.Depth).([]float64)
		(*e).insize = struct {
			x int
			y int
		}{x: len(in[0][0]), y: len(in[0])}
		(*e).input = in
		(*e).init = true
	}

	out = [][][]float64{}

	for o := 0; o < e.Depth; o++ {
		for i := 0; i < e.inputdepth; i++ {
			conv, err := math.Convolve(in[i], e.kernels[i][o], e.Stride)
			if err != nil {
				panic(err)
			}
			if len(out) - 1 < o {
				out = append(out, conv)
			} else {
				added, err := math.MatAdd(out[o], conv)
				if err != nil {
					panic(err)
				}
				out[o] = added
			}
		}

		out[o] = math.MatOp(out[o], func(val float64, x int, y int) float64 {
			return val + e.bias[o]
		})
	}

	return out
}

func (e *Conv2D) Backward(error [][][]float64, lr float64) [][][]float64 {
	//dB := []float64{}
	//dW := [][][]float64{}
	out := [][][]float64{}


	for o := 0; o < e.Depth; o++ {
		for i := 0; i < e.inputdepth; i++ {
			conv, err := math.Convolve(math.Padding(error[o], e.KernelSize.X-1),e.kernels[i][o], e.Stride)


			if err != nil {
				panic(err)
			}
			if i == 0 {
				out = append(out, conv)
			} else {
				added, err := math.MatAdd(out[o], conv)
				if err != nil {
					panic(err)
				}
				out[i] = added
			}

			// Update kernels
			conv, err = math.Convolve(e.input[i], error[o], 1)

			if err != nil {
				panic(err)
			}

			conv = math.MatOp(conv, func(val float64, x int, y int) float64 {
				return val * lr
			})

			updated, err := math.MatSub(e.kernels[i][o], conv)

			if err != nil {
				panic(err)
			}

			(*e).kernels[i][o] = updated
		}

		// Update bias
		(*e).bias[0] -= lr * (float64(e.Depth) * math.MatSum(error[o]))
	}

	return out
}

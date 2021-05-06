package layers

type Conv2D struct {
	KernelSize	[]float64
	Depth		[]float64
	kernels 	[][][]float64
	bias		[]float64
	init 		bool
}



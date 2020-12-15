package cnn

type Model struct {
	Name    string
	Layers  []interface{}
	Filters []float64 `json: "omitempty"`
}

type TypeName []interface{}
type Layer struct {
	L []interface{}
}

func (e Model) Forward(in [][]float64) ([][]float64, error) {
	current := in
	//&Model{Filters: []}
	for _, x := range e.Layers {
		if layer, ok := x.(MaxPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(AvgPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(Flatten); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, err
			} else {
				current = [][]float64{out}
			}
		}
		if layer, ok := x.(ReLu); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(Conv2D); ok {
			out, err := layer.Forward(current, []float64{1, 0, 1, 0})
			if err != nil {
				return nil, err
			} else {
				current = out
			}
		}
	}

	//MaxPooling{e.Config[0].(int), e.Config[1].(int)}.Forward(in)
	return current, nil
}

func (e Model) Fit(x_train [][][]float64, y_train []string, epochs int, learning_rate float64) {
	for i := 0; i < epochs; i++ {
		Forward()
	}
}

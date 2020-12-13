package cnn

type Model struct {
	Name   string
	Layers []interface{}
}

type TypeName []interface{}
type Layer struct {
	L []interface{}
}

func (e Model) Forward(in [][]float64) ([][]float64, error) {
	current := in

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
	}

	//MaxPooling{e.Config[0].(int), e.Config[1].(int)}.Forward(in)
	return current, nil
}

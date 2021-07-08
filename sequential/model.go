package sequential

import (
	"encoding/gob"
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/muesli/termenv"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Defines a sequential model
type Model struct {
	Name 	string
	Layers 	[]Layer
}

// Defines a layer with the functions for forward propagation and backward propagation
type Layer interface {
	Forward([][]float64) [][]float64
	Backward([][]float64, float64) [][]float64
}

// Defines a result object to analyze duration, error, acc etc.
type Result struct {
	Epochs 		int
	Duration 	time.Duration
	Error 		float64
	Accuracy	float64
}

// Reverses layers of the model
func reverse(s []Layer) []Layer {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
}

// Computes the input into output by forwarding it through all the layers
func (e Model) forward(in [][]float64) [][]float64 {
	for _, el := range e.Layers {
		in = (*(&el)).Forward(in)
	}
	return in
}

// Computes new weights for the layers by inputing the error of the output
func (e Model) backward(err [][]float64, lr float64) [][]float64 {
	reversed := reverse(e.Layers)
	for _, el := range e.Layers {
		err = (*(&el)).Backward(err, lr)
	}
	reversed = reverse(reversed)
	return err
}

// Adds a layer to the model
func (e *Model) Add(layer Layer) {
	e.Layers = append(e.Layers, layer)
}

// Decodes binary into a layer
func InterfaceDecode(decoder *gob.Decoder) Layer {

	var d Layer

	if err := decoder.Decode(&d); err != nil {
		log.Fatal(err)
	}

	return d

}

// Saves the model to a file
func (e *Model) Save(folder string, file ...string) error {
	if len(file) == 0 {
		file = append(file, "sequential-" + time.Now().Format("2006-01-02T15:04:05-MST"))
	}
	file[0] += ".bin"

	outdir, err := filepath.Abs(folder)

	if err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(outdir,file[0]))
	if err != nil {
		log.Fatal("Couldn't open file")
	}
	defer f.Close()

	//Register types
	gob.Register(layers.FC{})
	gob.Register(activation.Tanh{})

	layercount := len(e.Layers)

	enc := gob.NewEncoder(f)

	enc.Encode(layercount)

	for _,layer := range e.Layers {
		err = enc.Encode(layer)
		if err != nil {
			return err
		}
	}

	return err
}

// Let the model compute an output for the given input
func (e Model) Predict(in [][]float64) [][]float64 {
	return e.forward(in)
}

// Loads a model from a file
func Load(file string) (*Model,error) {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal("Couldn't open file")
	}
	defer f.Close()
	gob.Register(layers.FC{})
	gob.Register(activation.Tanh{})
	gob.Register(new(Layer))
	model := &Model{}
	dec := gob.NewDecoder(f)

	layercount := new(int)

	err = dec.Decode(layercount)
	fmt.Println(*layercount)
	if err != nil {
		return nil, err
	}
	for i := 0; i < *layercount; i++ {
		fmt.Println(i)
		var layer Layer
		layer = InterfaceDecode(dec)
		model.Layers = append(model.Layers, layer)
		if err != nil {
			return nil, err
		}
	}

	return model, nil
}

// Trains the network
func (e Model) Fit(x_train [][][]float64, y_train [][][]float64, epochs int, lr float64) Result {
	// Times the execution time
	start := time.Now()
	res := Result{Epochs: epochs}
	// Train for an amount of epochs
	for i := 1; i < epochs+1; i++ {
		er := 0.0
		avg := []float64{}
		// Forward and backward propagate all the objects through the model
		for ie, x := range x_train {
			out := e.forward(x)
			disperr, err := math2.MatriSubs(y_train[ie],out)
			if err != nil {
				fmt.Println(err)
			}
			// Squares all the values in the outputut error
			for y,_ := range disperr {
				for x,_ := range disperr[0] {
					disperr[y][x] *= disperr[y][x]
				}
			}
			// Calculates the mean of the error
			dispavg := []float64{}
			for _,row := range disperr {
				dispavg = append(dispavg, math2.Mean(row))
			}
			er += math2.Mean(dispavg)
			
			// Checks if the output is closest to the true value
			avg = append(avg, math2.Closest(out, y_train, y_train[ie]))

			// Calculates the actual error by substracting the output with the true value
			realerr, err := math2.MatriSubs(out, y_train[ie])
			if err != nil {
				fmt.Println(err)
			}
			for y,_ := range realerr {
				for x,_ := range realerr[0] {
					realerr[y][x] *= 2
				}
			}
			
			// Backpropagonate the error through the network
			e.backward(realerr, lr)

			Print(i, epochs, ie + 1, len(x_train), math2.Mean(avg), er)
		}
		acc := math2.Mean(avg)

		
		res.Accuracy = acc
		res.Error = er
	}
	
	end := time.Since(start)
	res.Duration = end

	return res
}

// Prints the output
func Print(epoch int, epochs int, item int, items int, acc float64, er float64)  {
	prog, err := progress.NewModel(progress.WithDefaultScaledGradient(),progress.WithoutPercentage(),)
	if err != nil {
		panic(err)
	}

	if epoch != 1 || item != 1 {
		termenv.ClearLines(6)
	}
	fmt.Println("")
	fmt.Println("Error:\t\t", er/float64(items))
	fmt.Println("Accuracy:\t",(acc*100),"%")
	fmt.Print("Items	")
	itemBar := termenv.String(prog.View(float64(item)/float64(items))).Foreground(termenv.ANSIMagenta)
	fmt.Print(strings.Replace(itemBar.String(),"░",termenv.String("█").Foreground(termenv.ANSIWhite).String(),-1))
	fmt.Println("\t",item,"\t/\t",items)
	fmt.Print("Epochs	")
	epochBar := termenv.String(prog.View(float64(epoch)/float64(epochs))).Foreground(termenv.ANSIBrightMagenta)
	fmt.Print(strings.Replace(epochBar.String(),"░",termenv.String("█").Foreground(termenv.ANSIBrightWhite).String(),-1))
	fmt.Println("\t",epoch,"\t/\t",epochs)
	fmt.Println("")
}

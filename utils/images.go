package utils

import (
	"image"
	"image/color"
	"image/jpeg"
	"os"
)

func ReadImage(path string) ([][]float64,error) {
	i, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer i.Close()
	img, err := jpeg.Decode(i)
	if err != nil {
		panic(err)
	}
	out := [][]float64{}
	bounds := img.Bounds()
	for y := 0; y < bounds.Dy(); y++  {
		row := []float64{}
		for x := 0; x < bounds.Dx(); x++  {
			r,g,b,_ := img.At(x,y).RGBA()
			//fmt.Println(r)
			// magic num 65535
			col := 1-(((float64(r)+float64(g)+float64(b))/3)/65535)
			//fmt.Println(col)
			row = append(row,col)
		}
		out = append(out, row)
	}
	return out, nil
}

func SaveImage(input [][]float64, path string) error {
	img := image.NewRGBA(image.Rect(0,0,len(input[0]), len(input)))

	for y:= 0; y < len(input); y++ {
		for x:= 0; x < len(input[0]); x++ {
			//fmt.Println()
			img.Set(x,y, color.RGBA{
				R: uint8(input[y][x]*255),
				G: uint8(input[y][x]*255),
				B: uint8(input[y][x]*255),
				A: 1,
			})
		}
	}
	file, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	err = jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
	if err != nil {
		panic(err)
	}
	return nil
}

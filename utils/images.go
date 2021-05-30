package utils

import (
	"github.com/TrizlyBear/PWS/math"
	"github.com/nfnt/resize"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"os"
)

type Decoder func(io.Reader) (image.Image, error)

func ReadImage(path string, dec Decoder) ([][]float64,error) {
	i, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer i.Close()
	img, err := dec(i)
	out := ImgToMat(img)
	return out, nil
}

func SaveImage(input [][]float64, path string) error {
	img := MatToImg(input)
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

func MatToImg(input [][]float64)  image.Image  {
	img := image.NewRGBA(image.Rect(0,0,len(input[0]), len(input)))

	for y := 0; y < len(input); y++ {
		for x := 0; x < len(input[0]); x++ {
			img.Set(x,y, color.RGBA{
				R: uint8(input[y][x]*255),
				G: uint8(input[y][x]*255),
				B: uint8(input[y][x]*255),
				A: 1,
			})
		}
	}
	return img
}

func ImgToMat(img image.Image) [][]float64 {
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
	return out
}

func Resize(x, y int,  img image.Image)  [][]float64 {
	resized := resize.Thumbnail(uint(x),uint(y),img,resize.Lanczos3)
	out := ImgToMat(resized)
	if y > resized.Bounds().Dy() {
		add := math.Zeros(len(out[0]), y-len(out))
		out = math.VertStack(out, add)
	}
	if x > resized.Bounds().Dx() {
		add := math.Zeros(x-len(out[0]), len(out))
		out = math.HorzStack(out, add)
	}
	return out
}

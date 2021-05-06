package utils

import "reflect"

func Inlist(list interface{}, item interface{}) bool {
	listVal := reflect.ValueOf(list)
	for i := 0; i < listVal.Len(); i++ {
		if listVal.Index(i).Interface() == item {
			return true
		}
	}
	return false
}

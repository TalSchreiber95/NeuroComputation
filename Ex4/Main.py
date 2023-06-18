from SOMGrid import KohonenGrid
from SOMLine import KohonenLine
from generateData import createDataA, createDataB

if __name__ == '__main__':
    grid = True
    type = "3 fingers"  # Part A = "Uniform" , "First non-uniform" , "Second non-uniform" , "Donut"
    # || Part B =  "4 fingers" , "3 fingers"
    neuronsSize = 200
    start = 0
    end = 1
    if type == "Donut":
        start = -4
        end = 4
        neuronsSize = 300
    data = 1
    if type == "Uniform" or type == "First non-uniform" or type == "Second non-uniform" or type == "Donut":
        data = createDataA(1000, type)  # "Uniform" , "First non-uniform" , "Second non-uniform" , "Donut"
    else:
        data = createDataB(type) # "4 fingers" , "3 fingers"
    if grid:
        model = KohonenGrid(data, neuronsSize , start , end)  # data , number of neurons , start , end
    else:
        model = KohonenLine(data, neuronsSize , start , end)  # data , number of neurons , start , end
    model.train_SOM(epochs=100)  # epochs=20 (epochs = num of iterations)

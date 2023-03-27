using Images

function readDucks()

  data = zeros(32*32, 50) # 128,128
  for nn=1:50
    filename = @sprintf("obj1__%d.png",nn)
    @printf("loading %s\n",filename)
    img = load(filename)
    data[:,nn]=vec(map(Float64,img[1:4:end,1:4:end]))
  end

  return data

end
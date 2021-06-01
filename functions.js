async function padMul128(){
  var canvas = document.getElementById("cInput");
  var ctx = canvas.getContext("2d");
  var image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let tensor = tf.browser.fromPixels(image)

  newH = Math.ceil(canvas.height/128.0)*128
  console.log("NEW HEIGHT: "+newH)
  newW = Math.ceil(canvas.width/128.0)*128
  console.log("NEW WIDTH: "+newW)
  console.log("newH-canvas.height:"+(newH-canvas.height))
  console.log("newW-canvas.width:"+(newW-canvas.width))
  padded = tensor.pad([[0, newH-canvas.height],[0,newW-canvas.width],[0,0]])
  console.log(padded.shape)
  padded.print()
  var cvv = document.createElement('canvas');
  cvv.id = "r3"
  cvv.width = padded.shape[0]
  cvv.height = padded.shape[0]
  await tf.browser.toPixels(padded, cvv);
  document.body.appendChild(cvv);
  console.log("UPSAMPLE128 END")
  slice()
}
async function slice(){
  var canvas = document.getElementById("r3");
  var ctx = canvas.getContext("2d");
  var image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let tensor = tf.browser.fromPixels(image)
  const b = tf.scalar(255.0);
  tensor = tensor.toFloat().div(b)
  row = []
  for (var i = 0; i < canvas.width; i+=128){
    col = []
    for (var j = 0; j < canvas.height; j+=128){
      res = tensor.slice([j,i],[128,128])
      res = await upsampleTo256(res)
      console.log("RES SHAPE SHOULD BE RANK 3:"+res.shape)
      col.push(res)
    }
    var fullcolumn = tf.concat(col,0)
    row.push(fullcolumn)
  }
  var complete = tf.concat(row,1)
  var cvv = document.createElement('canvas');
  cvv.width = res.shape[0]
  cvv.height = res.shape[0]
  console.log(complete.shape)
  await tf.browser.toPixels(complete.clipByValue(0,1), cvv);
  document.body.appendChild(cvv);
}
function removeElement(id) {
    var elem = document.getElementById(id);
    return elem.parentNode.removeChild(elem);
}
async function upsampleTo256(tensor) {
    const model = await tf.loadLayersModel('modelloh5/model.json');
    var res = []
    for(var i = 0; i<3; i++)
    {
      var layer = tf.unstack(tensor, 2)[i].expandDims(0).expandDims(-1).toFloat()
      var prediction = model.predict(layer).squeeze();//squeezes third axis
      res.push(prediction)
    }
    res = tf.stack(res,2)
    return res
}

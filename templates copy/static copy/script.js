


function convLayers(){
  let max = $("#max_conv")[0].value;
  for(let i=1;i<=max;i++){
    $("#convLayers").append("<h2><b>Conv layer "+i+"</b></h2><div style='display: inline-block'><div style='float:left;display:inline;margin:30px;width:300px;'>"+
    "<h1>No. of filters</h1> <input type='text' name='conv_filter"+[i]+"' required></div><div style='float:left;display:inline;margin:30px;width:300px;'>"+
      "<h1>Enter size of filter here</h1><input type='text' name='conv_size"+[i]+"'required > </div><div style='clear: both;'></div></div>")
      console.log($("conv_filter"+i))
      console.log($("conv_size"+i))
  }
}
  // <div style='float:left;display:inline;margin:30px;width:200px;'>
  //   <h1>No. of filters</h1>
  //   <input type="text" name="conv_filters">
  // </div>
  // <div style="float:left;display:inline;margin:30px;width:200px;">
  //   <h1>Enter size of filter here</h1>
  //   <input type='text' name='conv_size' placeholder = 'Please enter as ex. 1, 1'>
  // </div>
$("#convInput").click(convLayers);

function resetConv(){
  $( "#convLayers" ).empty();
}
$("#resetConv").click(resetConv)

function poolLayers(){
  let max = $("#max_pool")[0].value;
  for(let i=1;i<=max;i++){
    $("#poolLayers").append("<h2><b>Pool layer "+i+"</b></h2><div style='display: inline-block'><div style='float:left;display:inline;margin:-10px;width:300px;'>"+
    "<h1>Type of pooling</h1><select name='pool_type"+[i]+"'required >"+
    "<option value = 'Please select one'>Click</option>"+
    "<option value='MaxPooling2D'>MaxPooling2D</option><option value='AveragePooling2D'>AveragePooling2D</option>"+
    "</select></div><div style='float:left;display:inline;margin:-10px;width:300px;'>"+
      "<h1>Pooling size</h1><input type='text' name='pool_size"+[i]+"'required> </div><div style='clear: both;'></div></div><br>")
      console.log($("pool_type"+i))
      console.log($("pool_size"+i))
  }
}
$("#poolInput").click(poolLayers);

function resetPool(){
  $( "#poolLayers" ).empty();
}
$("#resetPool").click(resetPool)
function denseLayers(){
  let max = $("#max_dense")[0].value;
  for(let i=1;i<=max;i++){
    $("#denseLayers").append("<h2><b>Dense layer "+i+"</b></h2><div style='display: inline-block'><div style='float:left;display:inline;margin:-10px;width:300px;'>"+
    "<h1>No. of nodes</h1> <input type='text' name='dense_nodes"+[i]+"' required></div><div style='float:left;display:inline;margin:-10px;width:300px;'>"+
      "<h1>Activation Function</h1><select name='dense_activation"+i+"'required>"+
      "<option value = 'Please select one'>Click to select </option>"+
      "<option value='relu'>relu</option>"+
      "<option value='sigmoid'>sigmoid</option>"+
      "<option value='tanh'>tanh</option>"+
      "<option value='softplus'>softplus</option>"+
      "<option value='hard_sigmoid'>hard_sigmoid</option>"+
      "<option value='softmax'>softmax</option>"+
    "</select></div> <div style='clear: both;'></div></div><br>")
  }

}
$("#denseInput").click(denseLayers);

function resetDense(){
  $("#denseLayers").empty();
  console.log('hi')
}
$("#resetDense").click(resetDense);
// let x=1
// function prelayers(){
//   x+=1
//   $("#prelayers").append("<div><div style='display: inline-block;'> <div style='float:left;display:inline;margin:5px;'>"+
//       "<select id='preprocess' name='preprocess"+x+"'style = 'width:300px' required>"+
//         "<option>Click to select</option><option value = 'width_shift_range'>Width shift</option>"+
//         "<option value = 'height_shift_range'>Height shift</option>"+
//         "<option value = 'rotation_range'>Rotation</option><option value = 'horizontal_flip'>Horizontal flip</option>"+
//         "<option value = 'vertical_flip'>Vertical flip</option>"+
//         "<option value = 'Zoom'>Zoom</option></select></div><div style='float:left;display:inline;margin:5px;'>"+
//     "<input type='text' name'range"+x+"' placeholder='range' style = 'width:100px;'></div><div style='clear: both;'></div></div></div>")
// }
// $("#add").click(prelayers)
$( document ).ready(function(){
  if ($("#archi").text().length > 0){
    document.getElementById('archi').dispatchEvent(new MouseEvent("click"));
    document.getElementById('graph').dispatchEvent(new MouseEvent("click"));
  }
});

function resetImg(){
  $( "#test" ).empty();
  $( "#predictResult" ).empty();

  console.log('hi')
};
$("#img").click(resetImg);

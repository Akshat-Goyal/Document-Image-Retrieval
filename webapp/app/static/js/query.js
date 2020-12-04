function showIMG(data) {
    console.log(data);
    $("#imgdiv").empty();
    var main = document.getElementById("imgdiv");


    for(var i = 0; i < data.images.length; ++i) {
        if( i % 3==0 )
        {
            var row = document.createElement("div");
            row.className = "row";
            main.appendChild(row);
        }

        var image = " <img src=\"/static/image/".concat(data.images[i], "\" height=\"500px\" width=\"250px\"> ");
        var col = '<div class="column m-4"/> ' + image + "<br> <p>" + data.images[i] + "  <br/> dist:" + data.distances[i] + "</p>" +' <div/>'
        row.innerHTML = row.innerHTML +  col;
    }

}
function submit1() {
    document.getElementById("inpimg").style.display = "block";
    var filename = $("#iname").val();
    $('#inpimg') .attr('src', "/static/image/"+filename)
    if(filename){
        $.ajax({
            data: {
                filename : filename,
            },
            type : 'POST',
            url : '/search'
        })
            .done(function(data){
                showIMG(data);
            });
        event.preventDefault();
    }
}

function submit2() {
    document.getElementById("inpimg").style.display = "block";
    var data = new FormData();
    $.each(jQuery('#fileupload')[0].files, function(i, file) {
        data.append('file-'+i, file);
    });
    jQuery.ajax({
        url: '/searchUpload',
        data: data,
        cache: false,
        contentType: false,
        processData: false,
        method: 'POST',
        type: 'POST', // For jQuery < 1.9
        success: function(data){
            showIMG(data);
        }
    });



}
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#inpimg')
                .attr('src', e.target.result)
                .width(150)
                .height(200);
        };

        reader.readAsDataURL(input.files[0]);
    }
}


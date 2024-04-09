$("#loading").hide()




$("#upload_btn").on("click", function(e) {
    if($("#id_video").val()){
        // e.preventDefault();
        $("#input").hide()
        $(".loader_uploading").hide()
        $(".loader_processing").hide()
        
        $("#loading").show()
        $(".loader_uploading").show()
        setTimeout(() => {
            $(".loader_uploading").hide()
            $(".loader_processing").show()
        }, 3000);
    }
});

$("#report").on("click", function(e) {
    e.preventDefault();
    $("#report_btn").prop("disabled", true);
        path = $("#video_path").val();
        name = $("#name").val();
        pnum = $("#pnum").val();
        email = $("#email").val();
        complaint = $("#complaint").val();
    $.ajax({
        type: "GET",
        url: "/result",
        dataType: 'json',
        data: {action:'report',path,name,pnum,email,complaint},
        success:function(result) {
          console.log(result);
          Swal.fire({
            title: "E-Mail Sent Successfully!",
            icon: "success",
            toast:true,
            position:'top-end',
            timer:5000,
            timerProgressBar:true,
            showConfirmButton:false,
          });
          $("#report_btn").prop("disabled", false);
        },
        error:function(result) {
          console.log('error');
          Swal.fire({
            title: "Failed to sent E-Mail!",
            icon: "error",
            toast:true,
            position:'top-end',
            timer:5000,
            timerProgressBar:true,
            showConfirmButton:false,
          });
          $("#report_btn").prop("disabled", false);
        }
    });
});

$('#districts').change(()=>{
  var district = $("#districts").val();
  console.log(district);

})
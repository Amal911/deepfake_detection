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
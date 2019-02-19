var changeHeightWidth = function(){
    var img_width = $('#image-box').width();
    console.log(img_width);
    $('#image-box').height(img_width);
}
$(document).ready(changeHeightWidth);
$(window).resize(changeHeightWidth);
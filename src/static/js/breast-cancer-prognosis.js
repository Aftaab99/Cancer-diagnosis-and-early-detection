$(document).ready(function () {

    $('#result-heading').hide();
    $('#result-row').hide();

    var onCompletion = (res) => {
        var result;
        if(res.class==1){
            let prob = res.probability;
            $('#result').removeClass('text-success').addClass('text-danger');
            result=`Based on the features given, you are ${(prob*100).toFixed(2)}% likely to develop breast cancer.`
        }
        else{
            $('#result').removeClass('text-danger').addClass('text-success');
            result=`Based on the features given, you are unlikely to develop breast cancer`
        }
        $('#result').html(result);
        $('#result-row').show();
        $('#result-heading').show();
    }   

    $('form[name="breast-cancer-prognosis-form"]').submit(function (event) {
        event.preventDefault();
        var form = $('form[name="breast-cancer-prognosis-form"]')[0];
        if (form.checkValidity() === false) {
            console.log('Not valid!!')
            return;
        }
        else{
            let form_data = new FormData();
            console.log($('input[name="clump_thickness"]').val())
            console.log($('input[name="uniformity_cell_size"]').val())
            console.log($('input[name="uniformity_cell_shape"]').val())
            console.log($('input[name="marginal_adhesion"]').val())
            console.log($('input[name="single_epithelial_cell_size"]').val())
            console.log($('input[name="bare_nuclei"]').val())
            console.log($('input[name="bland_chromatin"]').val())
            console.log($('input[name="normal_nuclei"]').val())
            console.log($('input[name="mitosis"]').val())

            form_data.append('clump_thickness', $('input[name="clump_thickness"]').val());
            form_data.append('uniformity_cell_size', $('input[name="uniformity_cell_size"]').val());
            form_data.append('uniformity_cell_shape', $('input[name="uniformity_cell_shape"]').val());
            form_data.append('bland_chromatin', $('input[name="bland_chromatin"]').val());
            form_data.append('marginal_adhesion', $('input[name="marginal_adhesion"]').val());
            form_data.append('single_epithelial_cell_size', $('input[name="single_epithelial_cell_size"]').val());
            form_data.append('bare_nuclei', $('input[name="bare_nuclei"]').val());
            form_data.append('normal_nuclei', $('input[name="normal_nuclei"]').val());

            form_data.append('mitosis', $('input[name="mitosis"]').val());
            $.ajax({
                type: 'POST',
                data: form_data,
                processData: false,
                contentType: false,
                url: '/prognosis/breast_cancer'
            }).done(function(response){
                onCompletion(response);
            });
        }

    });

});
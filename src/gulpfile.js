var gulp = require('gulp');
var sass = require('gulp-sass');

gulp.task('sass', function () {
    return gulp.src(['node_modules/bootstrap/scss/bootstrap.scss', 'static/scss/*.scss'])
        .pipe(sass())
        .pipe(gulp.dest('static/css'));
});

gulp.task('js', function(){
    return gulp.src(['node_modules/bootstrap/dist/js/bootstrap.min.js', 'node_modules/jquery/dist/jquery.min.js', 'node_modules/tether/dist/js/tether.min.js'])
    .pipe(gulp.dest('static/js'));
});

gulp.task('default', gulp.parallel('sass', 'js'));
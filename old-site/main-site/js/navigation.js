function PopulateNavigationBarOnBlogPage(){
    console.log("started function");

    const navContent ='<ul><li><a href=\"http://henrysenior.com/index.html\">Home</a></li><li><a href=\"https://github.com/Delphboy\">GitHub</a></li><li><a href=\"https://www.linkedin.com/in/delphboy/\">LinkedIn</a></li><li><a href=\"http://henrysenior.com/blog.php\">Blog</a></li></ul>';

    let toc = document.getElementById("table-of-contents");
    let nav = document.createElement('nav');
    let line = document.createElement('hr');

    nav.innerHTML = navContent;
    toc.insertBefore(nav, toc.firstChild);
    nav.insertAdjacentHTML('afterend', '<hr />');
    console.log("ended function");
}

document.addEventListener("DOMContentLoaded", function(){
    PopulateNavigationBarOnBlogPage();
});

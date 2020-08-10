function PopulateFooter() {
    const footer = document.getElementById("footer");
    const date = new Date();
    const year = date.getFullYear();
    footer.innerHTML = `Colour theme: Solarized Light. &#169; Henry Senior ${year}. All Rights Reserved.`;
}

document.addEventListener("DOMContentLoaded", function()
{
    PopulateFooter();
});

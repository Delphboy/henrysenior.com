<!DOCTYPE HTML>
<html lang='en'>

<head>
    <link rel="stylesheet" type="text/css" href="css/main.css">

    <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=Edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description"
        content="Computer Science student at the University of Salford and developer at Carfinance 247" />
    <meta name="keywords" content="student, developer, salford university, Carfinance 247" />
    <meta property="og:title" content="Henry Senior | Computer Science Student and Developer" />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="http://www.henrysenior.com/" />

    <title>Henry Senior | Computer Scientist and Developer</title>
</head>

<body>
    <div>
        <h1>Blog</h1>
    </div>

    <nav>
        <ul>
            <li><a href="index.html">Home</a></li>
            <li><a href="https://github.com/Delphboy">GitHub</a></li>
            <li><a href="https://www.linkedin.com/in/delphboy/">LinkedIn</a></li>
            <li><a href="blog.php">Blog</a></li>
        </ul>
    </nav>
    <hr />
    <section>
         <?php
         $directory = "blog/";

// Open a directory, and read its contents
if (is_dir($directory)){
    if ($opendirectory = opendir($directory)){
        while (($file = readdir($opendirectory)) !== false){

            if(strpos($file, ".html") && !(strpos($file, "template"))){
                list($name, $extension) = explode(".", $file);
                echo "<li><a href='blog/" . $file . "'>" . $name . "</a></li>\n";
            }

        }
        closedir($opendirectory);
    }
}
         ?>

    </section>

    <footer id='footer'>
        &#169; Henry Senior
    </footer>
</body>

<script src="js/main.js"></script>

</html>

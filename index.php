<?php
header('Content-Type: application/text');

header('Content-Disposition: attachement; filename="embed30.txt"');

readfile('embed30.txt');
?>

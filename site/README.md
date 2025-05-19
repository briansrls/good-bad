# Frontend Showcase for @definitely-not-agi

This site serves as a frontend showcase and outreach platform for the concepts explored in the `definitely-not-agi` project. It uses modern web technologies (HTML, CSS, JavaScript with p5.js) to create interactive visualizations and explanations.

## Structure

-   `index.html`: The main entry point of the website.
-   `style.css`: Contains all the styling for the website, aiming for a modern look and feel.
-   `script.js`: Handles the interactive logic, including graphics rendering using the p5.js library. Currently features a dynamic fractal tree.

## Features

-   **Interactive Canvas**: Powered by p5.js for drawing graphics. The current example is a recursive fractal tree.
-   **Dynamic Control**: A slider allows users to adjust the detail (recursion depth) of the fractal.
-   **Mouse Interaction**: The angle of the fractal branches subtly changes based on horizontal mouse position over the canvas.
-   **Responsive Design**: The layout and canvas attempt to adapt to different screen sizes.
-   **Concept Explanations**: A dedicated section in `index.html` to explain the core ideas from the `definitely-not-agi` project, which can be expanded with more content and interactive demos.
-   **Modern UI**: Clean and responsive design with contemporary styling.

## Running the Site

1.  **No complex build process is required** for this simple setup.
2.  **Open `index.html` in a modern web browser** that supports JavaScript.
    -   You can usually do this by right-clicking the `index.html` file in your file explorer and choosing "Open with..." your preferred browser.
    -   Alternatively, you can navigate to its path directly in the browser, e.g., `file:///path/to/your/workspace/site/index.html` (replace with the actual path).

3.  **For best results and to avoid potential issues** with local file access restrictions (like CORS if external data sources are added later), it's highly recommended to serve the files through a simple local web server. Here are a couple of common ways:

    *   **Using Python's built-in HTTP server (Python 3.x):**
        ```bash
        # Navigate to the site directory in your terminal
        cd /root/good-bad/site 
        # (or your actual workspace path)/site

        # Start the server (default port is 8000)
        python3 -m http.server
        ```
        Then open `http://localhost:8000` (or `http://0.0.0.0:8000`) in your web browser.

    *   **Using Node.js with `npx` and `serve`:**
        If you have Node.js and npm installed, you can use `serve` without installing it globally:
        ```bash
        # Navigate to the site directory in your terminal
        cd /root/good-bad/site
        # (or your actual workspace path)/site

        # Start the server (it will tell you the port, often 3000)
        npx serve .
        ```
        Then open the URL provided by `serve` (e.g., `http://localhost:3000`) in your browser.

    *   **Using VS Code Live Server Extension:**
        If you are using VS Code, the "Live Server" extension is a convenient way to serve local HTML files.

## Development

-   **Graphics & Interactivity**: Modify `script.js` to change or add new p5.js sketches. You can create new functions for different visualizations (e.g., other types of fractals, data plots, interactive simulations).
-   **Content & Structure**: Edit `index.html` to add or modify text, sections, layout, and to integrate more demonstration areas.
-   **Styling**: Update `style.css` to change the appearance, add new styles for new elements, or refine the existing design.

## Future Enhancements Ideas

-   Integrate more specific visualizations directly tied to the `definitely-not-agi` project's outputs or core concepts.
-   Add more varied interactive controls (buttons, color pickers, input fields) for users to explore parameters of the visualizations.
-   Develop distinct "pages" or views for different concepts, potentially using simple client-side routing or by creating separate HTML files linked together.
-   Incorporate data loading and visualization if relevant to the project.
-   Expand the textual explanations alongside the interactive elements. 
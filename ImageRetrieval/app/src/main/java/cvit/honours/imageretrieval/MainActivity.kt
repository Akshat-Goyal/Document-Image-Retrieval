package cvit.honours.imageretrieval

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.android.volley.Request
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import org.json.JSONObject
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val cameraButton: Button = findViewById(R.id.cameraOpener)
        cameraButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, 1);
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK) {
            val extras: Bundle? = data?.extras
            val imageBitmap: Bitmap? = extras?.get("data") as Bitmap?
            val capturedImageView: ImageView = findViewById(R.id.capturedImageView)

            capturedImageView.setImageBitmap(imageBitmap)

            val stream = ByteArrayOutputStream()
            imageBitmap?.compress(Bitmap.CompressFormat.PNG, 100, stream)
            val imgB64 = Base64.encodeToString(stream.toByteArray(), Base64.DEFAULT)

            val queue = Volley.newRequestQueue(this)
            val url = "http://10.42.0.1:5000/searchUploadb64"

            val body = HashMap<String, String>()
            body["image"] = imgB64

            val request = JsonObjectRequest(
                    Request.Method.POST,
                    url,
                    JSONObject(body as Map<*, *>),
                    { response -> Toast.makeText(this, "Got result", Toast.LENGTH_SHORT).show() },
                    { error -> Toast.makeText(this, "An error occurred" + error.localizedMessage, Toast.LENGTH_LONG).show() }
            )

            queue.add(request)
        }
    }
}
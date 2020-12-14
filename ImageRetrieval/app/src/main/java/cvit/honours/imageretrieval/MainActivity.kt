package cvit.honours.imageretrieval

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.graphics.scale
import com.android.volley.DefaultRetryPolicy
import com.android.volley.toolbox.Volley
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.net.URL
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.HashMap
import kotlin.text.Charsets.UTF_8


class MainActivity : AppCompatActivity() {
    private lateinit var currentPhotoPath: String
    private lateinit var responseCounter: TextView
    private lateinit var fullImg: ImageView
    val serverURL = "http://10.42.0.1:5000"


    private val imgZoomListener = View.OnClickListener { view ->
        val imgView: ImageView = (view as ImageView)

        if (imgView.id == fullImg.id) {
            // remove full screen
            imgView.visibility = View.GONE
        } else {
            // set full screen
            fullImg.setImageBitmap((imgView.drawable as BitmapDrawable).bitmap)
            fullImg.visibility = View.VISIBLE
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val cameraButton: Button = findViewById(R.id.cameraOpener)
        val galleryButton: Button = findViewById(R.id.galleryOpener)

        fullImg = findViewById(R.id.fullImgView)
        fullImg.setOnClickListener(imgZoomListener)

        responseCounter = findViewById(R.id.responseCounter)

        cameraButton.setOnClickListener {
            responseCounter.text = ""
            Intent(MediaStore.ACTION_IMAGE_CAPTURE).also {
                val photoFile: File = createImageFile()
                val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.example.android.fileprovider",
                        photoFile
                )
                it.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(it, 1)
            }
        }

        galleryButton.setOnClickListener {
            responseCounter.text = ""
            Intent(Intent.ACTION_PICK).also {
                it.setType("image/*")
                startActivityForResult(it, 2)
            }
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File = getExternalFilesDir(Environment.DIRECTORY_PICTURES)!!
        return File.createTempFile(
                "PNG_${timeStamp}_",
                ".png",
                storageDir
        ).apply {
            currentPhotoPath = absolutePath
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        lateinit var img: Bitmap

        val capturedImageView: ImageView = findViewById(R.id.capturedImageView)
        capturedImageView.setOnClickListener(imgZoomListener)
        val imgContainer: LinearLayout = findViewById(R.id.imgContainer)


        if (requestCode == 1 && resultCode == RESULT_OK) {
            val extras: Bundle? = data?.extras
            val imageBitmap: Bitmap? = extras?.get("data") as Bitmap?

            capturedImageView.setImageBitmap(imageBitmap)

            img = BitmapFactory.decodeFile(currentPhotoPath)
            img = img.scale(img.width / 4, img.height / 4)
        } else if (requestCode == 2 && resultCode == RESULT_OK) {
            val uri = data?.data
            img = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

            capturedImageView.setImageBitmap(img)
        }

        val stream = ByteArrayOutputStream()
        img.compress(Bitmap.CompressFormat.PNG, 100, stream)

        val url = "${serverURL}/searchUpload"

        val request = object : VolleyFileUploadRequest(
                Method.POST,
                url,
                { response ->
                    val responseJSON = JSONObject(String(response.data, UTF_8))
                    val imgs = responseJSON.getJSONArray("images")

                    for (i in 0 until imgs.length()) {
                        Thread {
                            val imgName = imgs.getString(i)
                            val imgURL = URL("${serverURL}/static/image/${imgName}")
                            val bmp: Bitmap? = BitmapFactory.decodeStream(imgURL.openConnection().getInputStream())

                            runOnUiThread {
                                val imgView = ImageView(this)
                                imgView.setOnClickListener(imgZoomListener)
                                imgView.setImageBitmap(bmp)
                                imgContainer.addView(imgView)
                            }
                        }.start()
                    }

                    responseCounter.text = "Got ${imgs.length()} results"
                },
                { error -> Toast.makeText(this, "An error occurred: $error", Toast.LENGTH_LONG).show() }
        ) {
            override fun getByteData(): MutableMap<String, FileDataPart> {
                val params = HashMap<String, FileDataPart>()
                params["file-0"] = FileDataPart("image", stream.toByteArray(), "png")
                return params
            }
        }

        request.retryPolicy = DefaultRetryPolicy(
                60000, // 1m
                0,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        )

        Volley.newRequestQueue(this).add(request)
    }
}
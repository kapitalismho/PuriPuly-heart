use super::manifest::PhotoError;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Query, D3D11_ASYNC_GETDATA_DONOTFLUSH, D3D11_QUERY,
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT, D3D11_QUERY_DESC, D3D11_QUERY_TIMESTAMP,
    D3D11_QUERY_TIMESTAMP_DISJOINT,
};

pub(crate) struct GpuTimer {
    disjoint: ID3D11Query,
    start: ID3D11Query,
    end: ID3D11Query,
}

pub(crate) fn query(device: &ID3D11Device, kind: D3D11_QUERY) -> Result<ID3D11Query, PhotoError> {
    let mut query = None;
    unsafe {
        device.CreateQuery(
            &D3D11_QUERY_DESC {
                Query: kind,
                MiscFlags: 0,
            },
            Some(&mut query),
        )
    }
    .map_err(|_| PhotoError::Graphics)?;
    query.ok_or(PhotoError::Graphics)
}

impl GpuTimer {
    pub fn new(device: &ID3D11Device) -> Result<Self, PhotoError> {
        Ok(Self {
            disjoint: query(device, D3D11_QUERY_TIMESTAMP_DISJOINT)?,
            start: query(device, D3D11_QUERY_TIMESTAMP)?,
            end: query(device, D3D11_QUERY_TIMESTAMP)?,
        })
    }

    pub fn begin(&self, context: &ID3D11DeviceContext) {
        unsafe {
            context.Begin(&self.disjoint);
            context.End(&self.start);
        }
    }

    pub fn end(&self, context: &ID3D11DeviceContext) {
        unsafe {
            context.End(&self.end);
            context.End(&self.disjoint);
        }
    }

    pub fn elapsed_ms(&self, context: &ID3D11DeviceContext) -> Result<Option<f64>, PhotoError> {
        let mut disjoint = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
        let mut start = 0u64;
        let mut end = 0u64;
        unsafe {
            context
                .GetData(
                    &self.disjoint,
                    Some((&mut disjoint as *mut D3D11_QUERY_DATA_TIMESTAMP_DISJOINT).cast()),
                    std::mem::size_of_val(&disjoint) as u32,
                    D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
                .map_err(|_| PhotoError::Graphics)?;
            context
                .GetData(
                    &self.start,
                    Some((&mut start as *mut u64).cast()),
                    8,
                    D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
                .map_err(|_| PhotoError::Graphics)?;
            context
                .GetData(
                    &self.end,
                    Some((&mut end as *mut u64).cast()),
                    8,
                    D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
                .map_err(|_| PhotoError::Graphics)?;
        }
        if disjoint.Disjoint.as_bool() || disjoint.Frequency == 0 || end < start {
            return Ok(None);
        }
        Ok(Some(
            (end - start) as f64 * 1000.0 / disjoint.Frequency as f64,
        ))
    }
}
